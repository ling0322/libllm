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

//! Tests for the paged KV cache and the batch layout that addresses it.
//!
//! Both are the bookkeeping around attention rather than attention itself, so they run on the CPU
//! even though the attention kernels that read them do not.

use llm::flint::{DType, Device};
use llm::{ForwardBatch, KVCacheManager, KVCacheSpec};

fn spec() -> KVCacheSpec {
    KVCacheSpec {
        num_layers: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        max_context_length: 64,
        dtype: DType::Float,
    }
}

fn manager(block_size: i32, num_blocks: i32) -> KVCacheManager {
    KVCacheManager::new(spec(), block_size, num_blocks, Device::Cpu).unwrap()
}

#[test]
fn allocates_one_pool_per_layer() {
    let cache = manager(16, 4);

    assert_eq!(cache.num_blocks(), 4);
    assert_eq!(cache.block_size(), 16);
    for layer in 0..spec().num_layers {
        // Every layer addresses the same block ids in its own storage.
        assert_eq!(cache.key_cache(layer).unwrap().shape(), vec![4, 16, 2, 4]);
        assert_eq!(cache.value_cache(layer).unwrap().shape(), vec![4, 16, 2, 4]);
    }
    assert!(cache.key_cache(spec().num_layers).is_err(), "no such layer");
}

#[test]
fn hands_out_blocks_lowest_first_and_takes_them_back() {
    let mut cache = manager(16, 4);
    assert_eq!(cache.num_free_blocks(), 4);

    assert_eq!(cache.allocate_blocks(2).unwrap(), vec![0, 1]);
    assert_eq!(cache.allocate_blocks(1).unwrap(), vec![2]);
    assert_eq!(cache.num_free_blocks(), 1);

    cache.free_blocks(&[0, 1]).unwrap();
    assert_eq!(cache.num_free_blocks(), 3);

    // Freed blocks go back in, so the pool does not run down over time.
    let reused = cache.allocate_blocks(3).unwrap();
    assert_eq!(reused.len(), 3);
    assert_eq!(cache.num_free_blocks(), 0);

    assert!(
        cache.free_blocks(&[4]).is_err(),
        "block 4 is not in this pool"
    );
}

#[test]
fn allocation_is_all_or_nothing() {
    let mut cache = manager(16, 4);

    // A sequence that gets half of what it needs could not run, so nothing is taken.
    assert!(cache.allocate_blocks(5).is_none());
    assert_eq!(cache.num_free_blocks(), 4);

    assert_eq!(cache.allocate_blocks_for_tokens(17).unwrap(), vec![0, 1]);
    assert_eq!(cache.num_blocks_for_tokens(0), 0);
    assert_eq!(cache.num_blocks_for_tokens(16), 1);
    assert_eq!(cache.num_blocks_for_tokens(17), 2);
    assert_eq!(cache.max_num_blocks_per_request(), 4);
}

#[test]
fn refuses_a_block_size_that_is_not_a_power_of_two() {
    // The block of a position is found by shifting rather than dividing.
    let error = KVCacheManager::new(spec(), 12, 4, Device::Cpu).unwrap_err();
    assert!(error.to_string().contains("power of two"), "{error}");

    assert!(KVCacheManager::new(spec(), 16, 0, Device::Cpu).is_err());
}

#[test]
fn describes_a_single_sequence() {
    let batch = ForwardBatch::single(&[10, 11, 12], 5).unwrap();

    assert_eq!(batch.num_sequences(), 1);
    assert_eq!(batch.total_q_len(), 3);
    assert_eq!(batch.total_k_len(), 8, "the past counts towards the keys");
    assert_eq!(batch.max_q_len(), 3);
    assert_eq!(batch.max_k_len(), 8);
    assert_eq!(
        batch.position_ids(),
        &[5, 6, 7],
        "positions carry on from the past"
    );
    assert_eq!(batch.token_ids(), &[10, 11, 12]);
}

#[test]
fn describes_a_packed_batch() {
    let batch = ForwardBatch::packed(
        vec![1, 2, 3, 4],
        vec![0, 3, 4],
        vec![0, 3, 9],
        vec![0, 1, 2, 8],
    )
    .unwrap();

    assert_eq!(batch.num_sequences(), 2);
    assert_eq!(batch.total_q_len(), 4);
    assert_eq!(batch.max_q_len(), 3, "the first sequence is the longer one");
    assert_eq!(
        batch.max_k_len(),
        6,
        "the second one has the longer history"
    );

    // The lengths have to agree with the tokens, or the packing means something else.
    assert!(ForwardBatch::packed(vec![1, 2], vec![0, 3], vec![0, 3], vec![0, 1]).is_err());
    assert!(ForwardBatch::packed(vec![1], vec![0], vec![0], vec![0]).is_err());
}

#[test]
fn maps_every_token_to_its_slot_in_the_pool() {
    const BLOCK_SIZE: i32 = 4;

    // One sequence with 6 tokens of history, forwarding 3 more: positions 6, 7, 8 land at the end
    // of its second block and the start of its third.
    let mut batch = ForwardBatch::single(&[1, 2, 3], 6).unwrap();
    batch.set_block_ids(vec![vec![5, 2, 7]]).unwrap();
    let prepared = batch.prepare(Device::Cpu, BLOCK_SIZE).unwrap();

    assert_eq!(
        prepared.slot_mapping().to_vec_i32().unwrap(),
        vec![2 * BLOCK_SIZE + 2, 2 * BLOCK_SIZE + 3, 7 * BLOCK_SIZE],
    );
    assert_eq!(prepared.block_table().to_vec_i32().unwrap(), vec![5, 2, 7]);
    assert_eq!(prepared.cu_seqlens_q().to_vec_i32().unwrap(), vec![0, 3]);
    assert_eq!(prepared.seqlens_k().to_vec_i32().unwrap(), vec![9]);
    assert_eq!(
        prepared.token_ids().unwrap().to_vec_i64().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(prepared.position_ids().to_vec_i64().unwrap(), vec![6, 7, 8]);
    assert_eq!(prepared.last_query_indices().to_vec_i64().unwrap(), vec![2]);
}

#[test]
fn pads_the_block_table_to_the_longest_sequence() {
    let mut batch = ForwardBatch::packed(
        vec![1, 2, 3, 4],
        vec![0, 3, 4],
        vec![0, 3, 4],
        vec![0, 1, 2, 0],
    )
    .unwrap();
    batch.set_block_ids(vec![vec![0, 1], vec![2]]).unwrap();

    let prepared = batch.prepare(Device::Cpu, 2).unwrap();

    // The short row is padded; a kernel stops at that sequence's own length.
    assert_eq!(prepared.block_table().shape(), vec![2, 2]);
    assert_eq!(
        prepared.block_table().to_vec_i32().unwrap(),
        vec![0, 1, 2, 0]
    );
    assert_eq!(
        prepared.last_query_indices().to_vec_i64().unwrap(),
        vec![2, 3]
    );
    // Sequence 0 fills block 0 and spills into block 1; sequence 1 starts in its own block 2.
    assert_eq!(
        prepared.slot_mapping().to_vec_i32().unwrap(),
        vec![0, 1, 2, 4]
    );
}

#[test]
fn refuses_a_batch_that_is_not_ready() {
    let batch = ForwardBatch::single(&[1, 2], 0).unwrap();
    let error = batch.clone().prepare(Device::Cpu, 4).unwrap_err();
    assert!(error.to_string().contains("blocks"), "{error}");

    let mut wrong_count = batch.clone();
    assert!(wrong_count.set_block_ids(vec![vec![0], vec![1]]).is_err());

    // A sequence that outgrew its blocks would otherwise write over another one's.
    let mut outgrown = ForwardBatch::single(&[1, 2], 6).unwrap();
    outgrown.set_block_ids(vec![vec![0]]).unwrap();
    let error = outgrown.prepare(Device::Cpu, 4).unwrap_err();
    assert!(error.to_string().contains("past the"), "{error}");
}
