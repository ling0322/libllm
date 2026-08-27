// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! The cache of a model whose layers do not all need the same thing.

use llm::flint::{functional as F, DType, Device};
use llm::{
    CacheKind, FullAttentionSpec, KVCacheManager, KVCacheSpec, ModelCacheSpec, RecurrentStateSpec,
};

const NUM_LAYERS: i32 = 8;
const FULL_ATTENTION_INTERVAL: i32 = 4;
const MAX_CONTEXT: i32 = 256;

fn attention() -> FullAttentionSpec {
    FullAttentionSpec {
        num_key_value_heads: 2,
        head_dim: 4,
        dtype: DType::Float,
    }
}

/// Two value heads of four, and a convolution three wide over six channels: small enough to read,
/// the same shape as the layer it stands for.
fn recurrent() -> RecurrentStateSpec {
    RecurrentStateSpec::gated_delta_net(2, 4, 4, 6, 3, DType::Float).unwrap()
}

/// Three recurrent layers and then one attention layer, twice over.
fn hybrid() -> ModelCacheSpec {
    ModelCacheSpec::interleaved(
        NUM_LAYERS,
        FULL_ATTENTION_INTERVAL,
        attention(),
        recurrent(),
        MAX_CONTEXT,
    )
    .unwrap()
}

fn manager(num_blocks: i32, num_state_slots: i32) -> KVCacheManager {
    KVCacheManager::new(hybrid(), 16, num_blocks, num_state_slots, Device::Cpu).unwrap()
}

#[test]
fn the_attention_layers_are_the_last_of_each_group() {
    let spec = hybrid();

    assert_eq!(spec.num_layers(), NUM_LAYERS);
    assert_eq!(spec.layers_of(CacheKind::FullAttention), vec![3, 7]);
    assert_eq!(
        spec.layers_of(CacheKind::RecurrentState),
        vec![0, 1, 2, 4, 5, 6]
    );
    assert!(spec.has_recurrent_state());
}

#[test]
fn layers_that_need_the_same_thing_are_one_group() {
    let groups = hybrid().groups();

    // Two groups however far apart their layers run, and every layer in exactly one of them.
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].spec.kind(), CacheKind::RecurrentState);
    assert_eq!(groups[0].layers, vec![0, 1, 2, 4, 5, 6]);
    assert_eq!(groups[1].spec.kind(), CacheKind::FullAttention);
    assert_eq!(groups[1].layers, vec![3, 7]);

    // A model whose layers all agree is one group, which is what most families are.
    let uniform = ModelCacheSpec::uniform_attention(4, 2, 4, 64, DType::Float).unwrap();
    assert_eq!(uniform.groups().len(), 1);
    assert_eq!(uniform.groups()[0].layers, vec![0, 1, 2, 3]);
    assert!(!uniform.has_recurrent_state());
}

#[test]
fn a_page_costs_what_its_layers_hold() {
    let spec = hybrid();

    // A block holds the keys and the values of 16 tokens, for the two attention layers only.
    let per_layer = 2 * 16 * 2 * 4 * 4;
    assert_eq!(spec.bytes_per_block(16), 2 * per_layer);

    // A slot holds, for each of the six recurrent layers, the recurrence's (2, 4, 4) state and the
    // convolution's (6, 2) window.
    let per_slot = (2 * 4 * 4 + 6 * 2) * 4;
    assert_eq!(spec.bytes_per_state_slot(), 6 * per_slot);

    // What a request that runs to the end of the context costs is the two together, and neither
    // scales the way the other does: the blocks grow with the sequence and the slot does not.
    assert_eq!(spec.bytes_per_block(16) * 16, 2 * per_layer * 16);
}

#[test]
fn a_gated_delta_net_layer_carries_the_state_the_operator_reads() {
    let spec = recurrent();

    // The recurrence keeps one square state per value head, in float, which is what the slot
    // mapping of gated_delta_net_prefill indexes. The convolution keeps kernel - 1 inputs.
    assert_eq!(spec.shapes, vec![vec![2, 4, 4], vec![6, 2]]);
    assert_eq!(spec.dtypes, vec![DType::Float, DType::Float]);

    // A kernel of one would keep nothing, which is not a convolution this has to serve.
    assert!(RecurrentStateSpec::gated_delta_net(2, 4, 4, 6, 1, DType::Float).is_err());
    assert!(RecurrentStateSpec::gated_delta_net(0, 4, 4, 6, 3, DType::Float).is_err());
    assert!(RecurrentStateSpec::gated_delta_net(2, 4, 4, 0, 3, DType::Float).is_err());
    assert!(RecurrentStateSpec::gated_delta_net(2, 0, 4, 6, 3, DType::Float).is_err());
}

#[test]
fn each_kind_of_layer_gets_its_own_storage_and_only_its_own() {
    let cache = manager(4, 3);

    for layer in [3, 7] {
        assert_eq!(cache.key_cache(layer).unwrap().shape(), vec![4, 16, 2, 4]);
        assert_eq!(cache.value_cache(layer).unwrap().shape(), vec![4, 16, 2, 4]);
        // An attention layer has no state to carry between tokens.
        assert!(cache.state_cache(layer).is_err());
    }

    for layer in [0, 1, 2, 4, 5, 6] {
        let states = cache.state_cache(layer).unwrap();
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].shape(), vec![3, 2, 4, 4]);
        assert_eq!(states[1].shape(), vec![3, 6, 2]);
        // A recurrent layer attends over nothing, so it has no blocks.
        assert!(cache.key_cache(layer).is_err());
        assert!(cache.value_cache(layer).is_err());
    }

    assert!(cache.key_cache(NUM_LAYERS).is_err(), "no such layer");
    assert!(cache.state_cache(NUM_LAYERS).is_err(), "no such layer");
}

#[test]
fn a_slot_is_taken_whole_and_given_back_whole() {
    let mut cache = manager(4, 2);

    assert!(cache.needs_state_slot());
    assert_eq!(cache.num_state_slots(), 2);
    assert_eq!(cache.num_free_state_slots(), 2);

    // The lowest slot goes first, the way the lowest block does.
    let first = cache.allocate_state_slot().unwrap();
    let second = cache.allocate_state_slot().unwrap();
    assert_eq!((first, second), (0, 1));
    assert_eq!(cache.num_free_state_slots(), 0);

    // A request that cannot have a slot cannot run, and there is nothing partial to hand it.
    assert!(cache.allocate_state_slot().is_none());

    cache.free_state_slot(first).unwrap();
    assert_eq!(cache.num_free_state_slots(), 1);
    assert_eq!(cache.allocate_state_slot(), Some(first));

    // Freeing what is already free would hand the same slot to two requests.
    cache.free_state_slot(second).unwrap();
    assert!(cache.free_state_slot(second).is_err());
    assert!(cache.free_state_slot(-1).is_err());
    assert!(cache.free_state_slot(2).is_err());
}

#[test]
fn a_slot_comes_back_zeroed() {
    let mut cache = manager(4, 2);

    let slot = cache.allocate_state_slot().unwrap();
    for layer in [0, 4] {
        for state in cache.state_cache_mut(layer).unwrap() {
            let mut view = state.subtensor(slot).unwrap();
            F::fill(&mut view, 3.5).unwrap();
        }
    }
    cache.free_state_slot(slot).unwrap();

    // The next request to hold this slot starts from a zero state, not from whatever the last one
    // folded its history into. A recurrent layer reads its state before it writes it, so what is
    // left there is not merely stale, it is wrong.
    let reused = cache.allocate_state_slot().unwrap();
    assert_eq!(reused, slot);
    for layer in [0, 1, 2, 4, 5, 6] {
        for state in cache.state_cache(layer).unwrap() {
            let values = state.subtensor(reused).unwrap().to_vec_f32().unwrap();
            assert!(values.iter().all(|&x| x == 0.0), "layer {layer} is not zero");
        }
    }
}

#[test]
fn blocks_are_handed_out_as_before() {
    let mut cache = manager(4, 2);

    assert_eq!(cache.num_blocks(), 4);
    assert_eq!(cache.block_size(), 16);
    assert_eq!(cache.num_blocks_for_tokens(1), 1);
    assert_eq!(cache.num_blocks_for_tokens(16), 1);
    assert_eq!(cache.num_blocks_for_tokens(17), 2);
    assert_eq!(cache.max_num_blocks_per_request(), MAX_CONTEXT / 16);

    let blocks = cache.allocate_blocks_for_tokens(20).unwrap();
    assert_eq!(blocks, vec![0, 1]);
    assert_eq!(cache.num_free_blocks(), 2);

    // All or nothing: a sequence that got half of what it asked for could not run anyway.
    assert!(cache.allocate_blocks(3).is_none());
    assert_eq!(cache.num_free_blocks(), 2);

    cache.free_blocks(&blocks).unwrap();
    assert_eq!(cache.num_free_blocks(), 4);
    assert!(cache.free_blocks(&[4]).is_err());
}

#[test]
fn the_two_pools_are_sized_apart() {
    // Blocks and slots are counted separately, so a model can have many of one and few of the
    // other -- which is the point of not padding them to a common page size.
    let cache = manager(64, 2);
    assert_eq!(cache.num_blocks(), 64);
    assert_eq!(cache.num_state_slots(), 2);

    let cache = manager(2, 64);
    assert_eq!(cache.num_blocks(), 2);
    assert_eq!(cache.num_state_slots(), 64);
}

#[test]
fn a_model_without_recurrent_layers_has_no_slots() {
    let spec = ModelCacheSpec::uniform_attention(2, 2, 4, 64, DType::Float).unwrap();
    let mut cache = KVCacheManager::new(spec, 16, 4, 8, Device::Cpu).unwrap();

    // The slot count is ignored rather than honoured: there is nothing to put in them.
    assert!(!cache.needs_state_slot());
    assert_eq!(cache.num_state_slots(), 0);
    assert_eq!(cache.num_free_state_slots(), 0);
    assert!(cache.allocate_state_slot().is_none());
    assert!(cache.free_state_slot(0).is_err());
}

#[test]
fn a_model_with_recurrent_layers_needs_slots_to_be_built() {
    // Building it with none would leave every request unable to run, which is worth failing at
    // rather than discovering one request at a time.
    assert!(KVCacheManager::new(hybrid(), 16, 4, 0, Device::Cpu).is_err());

    // The block size is still a power of two, hybrid or not.
    assert!(KVCacheManager::new(hybrid(), 12, 4, 2, Device::Cpu).is_err());
}

#[test]
fn a_spec_has_to_describe_something_that_can_be_allocated() {
    assert!(ModelCacheSpec::new(Vec::new(), 64).is_err(), "no layers");
    assert!(
        ModelCacheSpec::uniform_attention(2, 2, 4, 0, DType::Float).is_err(),
        "no context"
    );
    assert!(
        ModelCacheSpec::uniform_attention(0, 2, 4, 64, DType::Float).is_err(),
        "no layers"
    );
    assert!(
        ModelCacheSpec::uniform_attention(2, 0, 4, 64, DType::Float).is_err(),
        "no heads"
    );
    assert!(
        ModelCacheSpec::interleaved(8, 0, attention(), recurrent(), 64).is_err(),
        "no interval"
    );

    // A state has to have a type, and a shape that describes a real tensor.
    let mismatched = KVCacheSpec::RecurrentState(RecurrentStateSpec {
        shapes: vec![vec![2, 2], vec![3, 3]],
        dtypes: vec![DType::Float],
    });
    assert!(ModelCacheSpec::new(vec![mismatched], 64).is_err());

    let empty = KVCacheSpec::RecurrentState(RecurrentStateSpec {
        shapes: vec![vec![2, 0]],
        dtypes: vec![DType::Float],
    });
    assert!(ModelCacheSpec::new(vec![empty], 64).is_err());
}

#[test]
fn an_interval_of_one_is_all_attention_and_a_long_one_is_none() {
    let all = ModelCacheSpec::interleaved(4, 1, attention(), recurrent(), 64).unwrap();
    assert_eq!(all.layers_of(CacheKind::FullAttention), vec![0, 1, 2, 3]);
    assert!(!all.has_recurrent_state());
    assert_eq!(all.bytes_per_state_slot(), 0);

    // An interval longer than the model leaves it with no attention layer at all, which is a
    // linear-attention-only model and needs no blocks.
    let none = ModelCacheSpec::interleaved(3, 8, attention(), recurrent(), 64).unwrap();
    assert!(none.layers_of(CacheKind::FullAttention).is_empty());
    assert_eq!(none.bytes_per_block(16), 0);

    let cache = KVCacheManager::new(none, 16, 0, 2, Device::Cpu).unwrap();
    assert_eq!(cache.num_blocks(), 0);
    assert_eq!(cache.num_state_slots(), 2);
}

/// The 27B, from `config.json` under `text_config`. Its `model_type` is `qwen3_5`, which is the
/// architecture the Qwen 3.8 models are built on.
fn qwen3_8_27b() -> ModelCacheSpec {
    qwen3_8_27b_in(DType::Float16)
}

/// The same, in whatever type the pools are to be allocated in. The model is bfloat16; the tests
/// that actually allocate ask for float, because the CPU backend only zeroes 16-bit tensors on
/// aarch64 and these run anywhere.
fn qwen3_8_27b_in(dtype: DType) -> ModelCacheSpec {
    ModelCacheSpec::qwen3_5(
        64,     // num_hidden_layers
        4,      // full_attention_interval
        4,      // num_key_value_heads
        256,    // head_dim
        16,     // linear_num_key_heads
        128,    // linear_key_head_dim
        48,     // linear_num_value_heads
        128,    // linear_value_head_dim
        4,      // linear_conv_kernel_dim
        262144, // max_position_embeddings
        dtype,
    )
    .unwrap()
}

#[test]
fn the_27b_caches_sixteen_of_its_sixty_four_layers() {
    let spec = qwen3_8_27b();

    // layer_types spells this out: three linear_attention and then one full_attention, sixteen
    // times over. Only those sixteen keep keys and values.
    assert_eq!(spec.num_layers(), 64);
    let attention = spec.layers_of(CacheKind::FullAttention);
    assert_eq!(attention.len(), 16);
    assert_eq!(&attention[..4], &[3, 7, 11, 15]);
    assert_eq!(attention[15], 63);
    assert_eq!(spec.layers_of(CacheKind::RecurrentState).len(), 48);

    // Two groups, however the layers interleave.
    assert_eq!(spec.groups().len(), 2);
}

#[test]
fn the_27b_costs_a_slot_what_it_costs_a_few_thousand_tokens_of_context() {
    let spec = qwen3_8_27b();

    // Keys and values: 16 layers, each 2 * 4 heads * 256 wide, in 16-bit. 64 KB a token, so a
    // block of 256 tokens is 16 MB.
    assert_eq!(spec.bytes_per_block(1), 64 * 1024);
    assert_eq!(spec.bytes_per_block(256), 16 * 1024 * 1024);

    // State: 48 layers, each a (48, 128, 128) recurrence in float and a (10240, 3) convolution
    // window in 16-bit. The recurrence is nearly all of it.
    let recurrence = 48 * 128 * 128 * 4;
    let convolution = (2 * 16 * 128 + 48 * 128) * 3 * 2;
    assert_eq!(convolution, 10240 * 3 * 2);
    assert_eq!(spec.bytes_per_state_slot(), 48 * (recurrence + convolution));
    assert_eq!(spec.bytes_per_state_slot(), 153_944_064); // 146.8 MiB

    // Which is the number max_num_seqs has to be chosen against: one request's state costs what
    // about 2300 tokens of its context do, and it is held whether the request is at its first
    // token or its last.
    let tokens_per_slot = spec.bytes_per_state_slot() / spec.bytes_per_block(1);
    assert_eq!(tokens_per_slot, 2349);

    // A hundred requests at once would be 15 GB of state before a single key is cached.
    assert_eq!(spec.bytes_per_state_slot() * 100 / (1024 * 1024 * 1024), 14);
}

#[test]
fn the_27b_pools_have_the_shapes_the_operators_read() {
    let spec = qwen3_8_27b_in(DType::Float);
    let cache = KVCacheManager::new(spec, 256, 2, 3, Device::Cpu).unwrap();

    // What flash attention pages through, and what gated_delta_net_prefill indexes by slot.
    assert_eq!(cache.key_cache(3).unwrap().shape(), vec![2, 256, 4, 256]);
    let states = cache.state_cache(0).unwrap();
    assert_eq!(states[0].shape(), vec![3, 48, 128, 128]);
    assert_eq!(states[1].shape(), vec![3, 10240, 3]);
    // The recurrence is float whatever the activations are -- mamba_ssm_dtype says float32 -- and
    // the convolution's window is in the activations' own type.
    assert_eq!(states[0].dtype(), DType::Float);
    assert_eq!(states[1].dtype(), DType::Float);
    assert_eq!(qwen3_8_27b().bytes_per_state_slot(), 153_944_064);
}
