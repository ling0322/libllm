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

//! The cache a model's layers need, for models whose layers do not all need the same thing.
//!
//! A full attention layer remembers every token it has seen, so what it needs grows with the
//! sequence and is handed out in blocks: a block holds the keys and values of a fixed number of
//! consecutive tokens, a block id names the same token range in every attention layer, and a
//! sequence owns a list of them. A linear attention layer — gated DeltaNet, and the short causal
//! convolution in front of it — remembers a fixed-size state instead, however long the sequence
//! gets, so what it needs is one slot, taken when the request starts and held until it finishes.
//!
//! Both are described the same way, by a [`KVCacheSpec`] per layer, which is the shape vLLM's
//! cache specs have: layers whose specs agree form a group, a group has one pool, and what a
//! request holds is one allocation per group. The two differ in exactly two places, and it is
//! worth naming them because everything else follows:
//!
//! - **How much a request needs.** An attention group's allocation grows a block at a time as the
//!   sequence does. A recurrent group's is one slot, always, decided when the request is admitted.
//! - **Whether it can be shared or rebuilt.** Attention blocks hold what was computed from a
//!   prefix, so two requests with the same prefix could share them. A recurrent state is the whole
//!   history folded together and cannot be cut at an arbitrary token, so a slot belongs to one
//!   request and a request that loses its slot has to start over.
//!
//! What this does not copy from vLLM is the padding that makes every group's page the same size.
//! vLLM needs it because one pool of uniform pages backs every group; here the pools are separate
//! tensors, so the recurrent pool is sized by how many requests may run at once and the attention
//! pool takes what is left, which wastes nothing.

use crate::flint::{functional as F, DType, Device, MemorySnapshot, Tensor};

use crate::engine_config::EngineConfig;
use crate::error::{Error, Result};
use crate::forward_batch::ForwardBatch;
use crate::model::ModelForGeneration;

/// What a full attention layer needs: the keys and values of every token, a block at a time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullAttentionSpec {
    /// The key-value heads of the layer, which is fewer than the query heads under grouped-query
    /// attention.
    pub num_key_value_heads: i32,
    /// The width of one head.
    pub head_dim: i32,
    /// The type the keys and values are stored in.
    pub dtype: DType,
}

impl FullAttentionSpec {
    /// The bytes one block of one layer takes, counting the key and the value.
    pub fn page_size_bytes(&self, block_size: i32) -> i64 {
        let elements =
            2 * i64::from(block_size) * i64::from(self.num_key_value_heads) * i64::from(self.head_dim);
        self.dtype.total_size(elements)
    }

    fn validate(&self) -> Result<()> {
        if self.num_key_value_heads <= 0 || self.head_dim <= 0 {
            return Err(Error::model(format!(
                "a full attention layer needs positive head counts, got {} heads of {}",
                self.num_key_value_heads, self.head_dim
            )));
        }
        Ok(())
    }
}

/// What a recurrent layer needs: a fixed-size state per sequence, whatever its length.
///
/// A layer may carry more than one, which is why this holds a list of shapes rather than one: a
/// gated DeltaNet layer carries the window of the causal convolution that feeds it as well as the
/// recurrence's own state, and the two have neither the same shape nor, in general, the same type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrentStateSpec {
    /// The shape of each state, without the slot dimension the pool adds in front.
    pub shapes: Vec<Vec<i32>>,
    /// The type of each state, one per shape.
    pub dtypes: Vec<DType>,
}

impl RecurrentStateSpec {
    /// The states a gated DeltaNet layer carries.
    ///
    /// The recurrence keeps `(numValueHeads, keyHeadDim, valueHeadDim)` — one state per value head,
    /// mapping the key space to the value space, which is what `F::gatedDeltaNetPrefill` reads and
    /// writes through its slot mapping. It stays in float, because it is a sum that a whole
    /// sequence accumulates into, and because that is what the models set `mamba_ssm_dtype` to.
    ///
    /// The convolution in front of it keeps the `kernelSize - 1` inputs of each channel that its
    /// next output still depends on, in the type the activations are in. Its width is not a head
    /// count: it convolves the queries, the keys and the values together, so the caller passes the
    /// total.
    pub fn gated_delta_net(
        num_value_heads: i32,
        key_head_dim: i32,
        value_head_dim: i32,
        conv_dim: i32,
        conv_kernel_size: i32,
        activation_dtype: DType,
    ) -> Result<RecurrentStateSpec> {
        if num_value_heads <= 0 || key_head_dim <= 0 || value_head_dim <= 0 {
            return Err(Error::model(format!(
                "a gated DeltaNet layer needs positive head counts, got {num_value_heads} heads \
                 of {key_head_dim} by {value_head_dim}"
            )));
        }
        if conv_dim <= 0 || conv_kernel_size <= 1 {
            return Err(Error::model(format!(
                "a gated DeltaNet layer's convolution needs a positive width and a kernel of at \
                 least two, got {conv_dim} and {conv_kernel_size}"
            )));
        }

        Ok(RecurrentStateSpec {
            shapes: vec![
                vec![num_value_heads, key_head_dim, value_head_dim],
                vec![conv_dim, conv_kernel_size - 1],
            ],
            dtypes: vec![DType::Float, activation_dtype],
        })
    }

    /// The bytes one slot of one layer takes, counting every state it carries.
    pub fn page_size_bytes(&self) -> i64 {
        self.shapes
            .iter()
            .zip(self.dtypes.iter())
            .map(|(shape, dtype)| dtype.total_size(shape.iter().map(|&d| i64::from(d)).product()))
            .sum()
    }

    fn validate(&self) -> Result<()> {
        if self.shapes.is_empty() {
            return Err(Error::model("a recurrent layer must carry at least one state"));
        }
        if self.shapes.len() != self.dtypes.len() {
            return Err(Error::model(format!(
                "a recurrent layer needs a type per state, got {} shapes and {} types",
                self.shapes.len(),
                self.dtypes.len()
            )));
        }
        for shape in &self.shapes {
            if shape.is_empty() || shape.iter().any(|&d| d <= 0) {
                return Err(Error::model(format!(
                    "a recurrent state's shape must be positive in every dimension, got {shape:?}"
                )));
            }
        }
        Ok(())
    }
}

/// What one layer needs of the cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KVCacheSpec {
    FullAttention(FullAttentionSpec),
    RecurrentState(RecurrentStateSpec),
}

impl KVCacheSpec {
    /// What kind of pool this layer draws from. Layers agree on their kind before they can be
    /// asked whether they agree on everything else.
    pub fn kind(&self) -> CacheKind {
        match self {
            KVCacheSpec::FullAttention(_) => CacheKind::FullAttention,
            KVCacheSpec::RecurrentState(_) => CacheKind::RecurrentState,
        }
    }

    /// The bytes one layer of this kind takes for one allocation of it: a block for an attention
    /// layer, a slot for a recurrent one.
    pub fn page_size_bytes(&self, block_size: i32) -> i64 {
        match self {
            KVCacheSpec::FullAttention(spec) => spec.page_size_bytes(block_size),
            KVCacheSpec::RecurrentState(spec) => spec.page_size_bytes(),
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            KVCacheSpec::FullAttention(spec) => spec.validate(),
            KVCacheSpec::RecurrentState(spec) => spec.validate(),
        }
    }
}

/// The kinds of pool a layer can draw from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheKind {
    FullAttention,
    RecurrentState,
}

/// The layers that share a spec, and so share a pool. vLLM calls this a KV cache group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KVCacheGroup {
    /// The layers in it, in the order the model runs them.
    pub layers: Vec<i32>,
    /// What each of them needs, which is the same for all of them.
    pub spec: KVCacheSpec,
}

/// The cache a whole model needs: what every layer needs, and how far back it may have to reach.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCacheSpec {
    layers: Vec<KVCacheSpec>,
    max_context_length: i32,
}

impl ModelCacheSpec {
    /// The spec of a model whose layers are given one by one, in the order it runs them.
    pub fn new(layers: Vec<KVCacheSpec>, max_context_length: i32) -> Result<ModelCacheSpec> {
        if layers.is_empty() {
            return Err(Error::model("a model needs at least one layer to cache"));
        }
        if max_context_length <= 0 {
            return Err(Error::model(format!(
                "max_context_length must be positive, got {max_context_length}"
            )));
        }
        for layer in &layers {
            layer.validate()?;
        }

        Ok(ModelCacheSpec {
            layers,
            max_context_length,
        })
    }

    /// The spec of a model whose layers are all the same full attention, which is what most model
    /// families are.
    pub fn uniform_attention(
        num_layers: i32,
        num_key_value_heads: i32,
        head_dim: i32,
        max_context_length: i32,
        dtype: DType,
    ) -> Result<ModelCacheSpec> {
        if num_layers <= 0 {
            return Err(Error::model(format!(
                "a model needs at least one layer, got {num_layers}"
            )));
        }

        let spec = KVCacheSpec::FullAttention(FullAttentionSpec {
            num_key_value_heads,
            head_dim,
            dtype,
        });
        ModelCacheSpec::new(vec![spec; num_layers as usize], max_context_length)
    }

    /// The spec of a model that runs a full attention layer every so often and a recurrent one the
    /// rest of the time, which is the shape of the Qwen3 hybrids: `full_attention_interval` of 4
    /// means three gated DeltaNet layers and then one attention layer, repeated.
    ///
    /// The attention layer is the last of each group rather than the first, which is the convention
    /// the Qwen configurations use — `(layer + 1) % interval == 0`. A model whose pattern is
    /// anything else should hand [`ModelCacheSpec::new`] the layers directly rather than describe
    /// it here.
    pub fn interleaved(
        num_layers: i32,
        full_attention_interval: i32,
        attention: FullAttentionSpec,
        recurrent: RecurrentStateSpec,
        max_context_length: i32,
    ) -> Result<ModelCacheSpec> {
        if num_layers <= 0 {
            return Err(Error::model(format!(
                "a model needs at least one layer, got {num_layers}"
            )));
        }
        if full_attention_interval <= 0 {
            return Err(Error::model(format!(
                "full_attention_interval must be positive, got {full_attention_interval}"
            )));
        }

        let layers = (0..num_layers)
            .map(|layer| {
                if (layer + 1) % full_attention_interval == 0 {
                    KVCacheSpec::FullAttention(attention)
                } else {
                    KVCacheSpec::RecurrentState(recurrent.clone())
                }
            })
            .collect();
        ModelCacheSpec::new(layers, max_context_length)
    }

    /// The spec of a Qwen3.5 text stack — the architecture the Qwen 3.8 models are built on, whose
    /// `model_type` is `qwen3_5` — from the fields of its configuration, under the names the
    /// configuration uses.
    ///
    /// Three gated DeltaNet layers and then one full attention layer, repeated, which is what
    /// `layer_types` spells out and `full_attention_interval` summarises. The convolution's width
    /// is not in the configuration: it convolves the queries, keys and values of the linear layers
    /// together, so it is `2 * key_dim + value_dim`.
    ///
    /// For the 27B: 64 layers, 16 of them attention with 4 key-value heads of 256, and 48 of them
    /// gated DeltaNet with 48 value heads of 128 by 128 over a convolution of 10240 channels and a
    /// kernel of 4. That is 64 KB of keys and values per token, against 147 MB of state per
    /// request — a slot costs what about 2300 tokens of context do, which is what `max_num_seqs`
    /// has to be chosen against.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen3_5(
        num_hidden_layers: i32,
        full_attention_interval: i32,
        num_key_value_heads: i32,
        head_dim: i32,
        linear_num_key_heads: i32,
        linear_key_head_dim: i32,
        linear_num_value_heads: i32,
        linear_value_head_dim: i32,
        linear_conv_kernel_dim: i32,
        max_position_embeddings: i32,
        dtype: DType,
    ) -> Result<ModelCacheSpec> {
        if linear_num_key_heads <= 0 || linear_key_head_dim <= 0 {
            return Err(Error::model(format!(
                "a gated DeltaNet layer needs positive key head counts, got \
                 {linear_num_key_heads} heads of {linear_key_head_dim}"
            )));
        }

        let key_dim = linear_num_key_heads * linear_key_head_dim;
        let value_dim = linear_num_value_heads * linear_value_head_dim;
        let recurrent = RecurrentStateSpec::gated_delta_net(
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            2 * key_dim + value_dim,
            linear_conv_kernel_dim,
            dtype,
        )?;

        ModelCacheSpec::interleaved(
            num_hidden_layers,
            full_attention_interval,
            FullAttentionSpec {
                num_key_value_heads,
                head_dim,
                dtype,
            },
            recurrent,
            max_position_embeddings,
        )
    }

    pub fn num_layers(&self) -> i32 {
        self.layers.len() as i32
    }

    /// The most tokens the model can attend over, which bounds what one sequence can own.
    pub fn max_context_length(&self) -> i32 {
        self.max_context_length
    }

    /// What one layer needs.
    pub fn layer(&self, layer: i32) -> Result<&KVCacheSpec> {
        self.layers
            .get(layer as usize)
            .ok_or_else(|| Error::model(format!("layer {layer} is not in the cache spec")))
    }

    /// The layers of one kind, in the order the model runs them.
    pub fn layers_of(&self, kind: CacheKind) -> Vec<i32> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, spec)| spec.kind() == kind)
            .map(|(layer, _)| layer as i32)
            .collect()
    }

    /// The groups the layers fall into: those that need the same thing share a pool. Layers that
    /// agree on everything but their position are one group however far apart they run.
    pub fn groups(&self) -> Vec<KVCacheGroup> {
        let mut groups: Vec<KVCacheGroup> = Vec::new();
        for (layer, spec) in self.layers.iter().enumerate() {
            match groups.iter_mut().find(|group| &group.spec == spec) {
                Some(group) => group.layers.push(layer as i32),
                None => groups.push(KVCacheGroup {
                    layers: vec![layer as i32],
                    spec: spec.clone(),
                }),
            }
        }
        groups
    }

    /// Whether any layer carries a state that has to be kept for a whole request.
    pub fn has_recurrent_state(&self) -> bool {
        self.layers
            .iter()
            .any(|spec| spec.kind() == CacheKind::RecurrentState)
    }

    /// The bytes one block takes across every attention layer, which is what a block of the pool
    /// costs.
    pub fn bytes_per_block(&self, block_size: i32) -> i64 {
        self.layers
            .iter()
            .filter(|spec| spec.kind() == CacheKind::FullAttention)
            .map(|spec| spec.page_size_bytes(block_size))
            .sum()
    }

    /// The bytes one slot takes across every recurrent layer.
    pub fn bytes_per_state_slot(&self) -> i64 {
        self.layers
            .iter()
            .filter(|spec| spec.kind() == CacheKind::RecurrentState)
            .map(|spec| spec.page_size_bytes(1))
            .sum()
    }
}

/// Owns the cache storage and hands out blocks and slots of it.
#[derive(Debug)]
pub struct KVCacheManager {
    spec: ModelCacheSpec,
    block_size: i32,
    num_blocks: i32,
    num_state_slots: i32,
    /// Indexed by layer, empty for layers that are not full attention.
    key_cache: Vec<Option<Tensor>>,
    value_cache: Vec<Option<Tensor>>,
    /// Indexed by layer, and then by the states that layer carries. Empty for layers that carry
    /// none.
    state_cache: Vec<Vec<Tensor>>,
    /// The blocks nobody holds, kept in descending order so that the lowest id is taken first.
    free_blocks: Vec<i32>,
    /// The state slots nobody holds, in the same order and for the same reason.
    free_state_slots: Vec<i32>,
}

impl KVCacheManager {
    /// Allocate the pools of every layer.
    ///
    /// `block_size` is a power of two, which is what lets a token's position be split into a block
    /// and an offset without a division at every step. `num_state_slots` is how many requests may
    /// hold a recurrent state at once, and so how many may run at once on a model that has one; it
    /// is ignored by a model that has no recurrent layer.
    pub fn new(
        spec: ModelCacheSpec,
        block_size: i32,
        num_blocks: i32,
        num_state_slots: i32,
        device: Device,
    ) -> Result<KVCacheManager> {
        if block_size <= 0 {
            return Err(Error::model(format!(
                "kv cache needs a positive block size, got {block_size}"
            )));
        }
        if block_size & (block_size - 1) != 0 {
            return Err(Error::model(format!(
                "kv cache block size must be a power of two, got {block_size}"
            )));
        }

        let num_attention_layers = spec.layers_of(CacheKind::FullAttention).len();
        if num_attention_layers > 0 && num_blocks <= 0 {
            return Err(Error::model(format!(
                "a model with attention layers needs a positive block count, got {num_blocks}"
            )));
        }
        let num_state_slots = if spec.has_recurrent_state() {
            if num_state_slots <= 0 {
                return Err(Error::model(format!(
                    "a model with recurrent layers needs a positive slot count, got \
                     {num_state_slots}"
                )));
            }
            num_state_slots
        } else {
            0
        };

        let mut key_cache = Vec::with_capacity(spec.layers.len());
        let mut value_cache = Vec::with_capacity(spec.layers.len());
        let mut state_cache = Vec::with_capacity(spec.layers.len());
        for layer in &spec.layers {
            match layer {
                KVCacheSpec::FullAttention(attention) => {
                    let shape = [
                        num_blocks,
                        block_size,
                        attention.num_key_value_heads,
                        attention.head_dim,
                    ];
                    // Every element is written by store_kv_cache() before attention reads it, so
                    // there is nothing to gain from zeroing what can be tens of gigabytes.
                    key_cache.push(Some(Tensor::empty(&shape, attention.dtype, device)?));
                    value_cache.push(Some(Tensor::empty(&shape, attention.dtype, device)?));
                    state_cache.push(Vec::new());
                }
                KVCacheSpec::RecurrentState(recurrent) => {
                    key_cache.push(None);
                    value_cache.push(None);

                    // A recurrent state is read before it is written -- the first chunk of a
                    // sequence folds the state it came in with into its own output -- so a slot
                    // that a request has just been handed has to be zero rather than whatever the
                    // request before it left there.
                    let mut states = Vec::with_capacity(recurrent.shapes.len());
                    for (shape, &dtype) in recurrent.shapes.iter().zip(recurrent.dtypes.iter()) {
                        let mut slot_shape = Vec::with_capacity(shape.len() + 1);
                        slot_shape.push(num_state_slots);
                        slot_shape.extend_from_slice(shape);
                        states.push(Tensor::zeros(&slot_shape, dtype, device)?);
                    }
                    state_cache.push(states);
                }
            }
        }

        Ok(KVCacheManager {
            spec,
            block_size,
            num_blocks,
            num_state_slots,
            key_cache,
            value_cache,
            state_cache,
            free_blocks: (0..num_blocks).rev().collect(),
            free_state_slots: (0..num_state_slots).rev().collect(),
        })
    }

    /// The key pool of one attention layer, `<dtype>(numBlocks, blockSize, numKeyValueHeads,
    /// headDim)`.
    pub fn key_cache(&self, layer: i32) -> Result<&Tensor> {
        self.key_cache
            .get(layer as usize)
            .and_then(|cache| cache.as_ref())
            .ok_or_else(|| Error::model(format!("layer {layer} is not a full attention layer")))
    }

    /// The value pool of one attention layer, shaped like the key pool.
    pub fn value_cache(&self, layer: i32) -> Result<&Tensor> {
        self.value_cache
            .get(layer as usize)
            .and_then(|cache| cache.as_ref())
            .ok_or_else(|| Error::model(format!("layer {layer} is not a full attention layer")))
    }

    /// The states of one recurrent layer, each `<dtype>(numSlots, ...)` and in the order the spec
    /// gave them. A gated DeltaNet layer has the recurrence's state first and the convolution's
    /// window second.
    pub fn state_cache(&self, layer: i32) -> Result<&[Tensor]> {
        match self.state_cache.get(layer as usize) {
            Some(states) if !states.is_empty() => Ok(states),
            _ => Err(Error::model(format!(
                "layer {layer} is not a recurrent layer"
            ))),
        }
    }

    /// The states of one recurrent layer, to write into.
    pub fn state_cache_mut(&mut self, layer: i32) -> Result<&mut [Tensor]> {
        match self.state_cache.get_mut(layer as usize) {
            Some(states) if !states.is_empty() => Ok(states),
            _ => Err(Error::model(format!(
                "layer {layer} is not a recurrent layer"
            ))),
        }
    }

    /// The key and value pools of one layer, to write into.
    ///
    /// Both at once, because a forward pass writes the keys and the values of a layer together and
    /// two separate borrows of the same manager would not be allowed.
    pub fn caches_mut(&mut self, layer: i32) -> Result<(&mut Tensor, &mut Tensor)> {
        let index = layer as usize;
        if index >= self.key_cache.len() || self.key_cache[index].is_none() {
            return Err(Error::model(format!(
                "layer {layer} is not a full attention layer"
            )));
        }

        let (keys, values) = (&mut self.key_cache[index], &mut self.value_cache[index]);
        match (keys.as_mut(), values.as_mut()) {
            (Some(keys), Some(values)) => Ok((keys, values)),
            _ => Err(Error::model(format!(
                "layer {layer} is not a full attention layer"
            ))),
        }
    }

    pub fn spec(&self) -> &ModelCacheSpec {
        &self.spec
    }

    /// The tokens one block holds.
    pub fn block_size(&self) -> i32 {
        self.block_size
    }

    pub fn num_blocks(&self) -> i32 {
        self.num_blocks
    }

    pub fn num_free_blocks(&self) -> i32 {
        self.free_blocks.len() as i32
    }

    /// How many requests may hold a recurrent state at once. Zero for a model without one.
    pub fn num_state_slots(&self) -> i32 {
        self.num_state_slots
    }

    pub fn num_free_state_slots(&self) -> i32 {
        self.free_state_slots.len() as i32
    }

    /// Whether a request has to hold a state slot to run at all.
    pub fn needs_state_slot(&self) -> bool {
        self.num_state_slots > 0
    }

    /// The blocks `num_tokens` tokens need.
    pub fn num_blocks_for_tokens(&self, num_tokens: i32) -> i32 {
        (num_tokens + self.block_size - 1) / self.block_size
    }

    /// The blocks a sequence that runs to the end of the context would need, which is what a
    /// scheduler reserves room for.
    pub fn max_num_blocks_per_request(&self) -> i32 {
        self.num_blocks_for_tokens(self.spec.max_context_length())
    }

    /// Take `num_blocks` blocks, or nothing at all when there are not that many left.
    ///
    /// All or nothing: a sequence that gets half of what it asked for could not run anyway, and
    /// leaving the pool untouched keeps a scheduler from having to hand blocks back.
    pub fn allocate_blocks(&mut self, num_blocks: i32) -> Option<Vec<i32>> {
        if num_blocks < 0 || num_blocks > self.num_free_blocks() {
            return None;
        }

        let at = self.free_blocks.len() - num_blocks as usize;
        let mut taken = self.free_blocks.split_off(at);
        // The free list runs high to low, so taking from its end gives the lowest ids; reversing
        // puts them back in ascending order, which is the order the sequence fills them in.
        taken.reverse();
        Some(taken)
    }

    /// Take the blocks `num_tokens` tokens need.
    pub fn allocate_blocks_for_tokens(&mut self, num_tokens: i32) -> Option<Vec<i32>> {
        self.allocate_blocks(self.num_blocks_for_tokens(num_tokens))
    }

    /// Give blocks back.
    pub fn free_blocks(&mut self, block_ids: &[i32]) -> Result<()> {
        for &block_id in block_ids {
            if block_id < 0 || block_id >= self.num_blocks {
                return Err(Error::model(format!(
                    "block {block_id} is not one of this cache's {} blocks",
                    self.num_blocks
                )));
            }
            self.free_blocks.push(block_id);
        }
        Ok(())
    }

    /// Take a state slot, or nothing when every one of them is held.
    ///
    /// A request takes one when it is admitted and holds it until it finishes: the state is the
    /// whole of its history folded together, so unlike a block it cannot be handed to somebody else
    /// and read back later. The slot comes back zeroed, which is the state a sequence starts from.
    pub fn allocate_state_slot(&mut self) -> Option<i32> {
        let slot = self.free_state_slots.pop()?;
        // Whatever the request before it left here is not a prefix of this one's history. Zeroing
        // on the way out rather than on the way in keeps a slot that is never reused from being
        // cleared twice, and there is nothing to read between the two.
        if let Err(error) = self.clear_state_slot(slot) {
            self.free_state_slots.push(slot);
            let _ = error;
            return None;
        }
        Some(slot)
    }

    /// Give a state slot back.
    pub fn free_state_slot(&mut self, slot: i32) -> Result<()> {
        if slot < 0 || slot >= self.num_state_slots {
            return Err(Error::model(format!(
                "state slot {slot} is not one of this cache's {} slots",
                self.num_state_slots
            )));
        }
        if self.free_state_slots.contains(&slot) {
            return Err(Error::model(format!("state slot {slot} is already free")));
        }
        self.free_state_slots.push(slot);
        Ok(())
    }

    /// Zero one slot of every recurrent layer, which is the state a sequence begins with.
    fn clear_state_slot(&mut self, slot: i32) -> Result<()> {
        for states in &mut self.state_cache {
            for state in states.iter_mut() {
                let mut view = state.subtensor(slot)?;
                F::fill(&mut view, 0.0)?;
            }
        }
        Ok(())
    }

    /// Build the cache a model should have, filling the device memory that is left once the
    /// weights and one full-size forward pass are accounted for.
    ///
    /// How much a pass takes is measured rather than guessed: a full-size batch is forwarded
    /// against a scratch pool just large enough to hold it, and the peak that follows is what the
    /// cache has to leave room for. Needs a device that reports its memory usage, which the CPU
    /// backend does not.
    ///
    /// The recurrent pool is sized first, at one slot per request that may run at once, because
    /// that count is what bounds the batch rather than something the cache gets to choose. The
    /// blocks then take whatever is left.
    pub fn for_model<M: ModelForGeneration>(
        model: &M,
        config: &EngineConfig,
    ) -> Result<KVCacheManager> {
        let spec = model.kv_cache_spec()?;
        let block_size = config.kv_cache_block_size;
        let num_state_slots = if spec.has_recurrent_state() {
            config.max_num_seqs
        } else {
            0
        };

        let budget = Self::estimate_memory_budget(model, config, &spec, num_state_slots)?;
        let state_bytes = spec.bytes_per_state_slot() * i64::from(num_state_slots);
        let bytes_per_block = spec.bytes_per_block(block_size);

        let num_blocks = if bytes_per_block > 0 {
            let left = budget - state_bytes;
            if left > 0 {
                (left / bytes_per_block).min(i64::from(i32::MAX)) as i32
            } else {
                0
            }
        } else {
            // A model with no attention layer at all needs no blocks, and asking for none is not
            // an error there.
            0
        };
        if bytes_per_block > 0 && num_blocks <= 0 {
            return Err(Error::model(
                "not enough device memory left for even one block of the kv cache",
            ));
        }

        KVCacheManager::new(spec, block_size, num_blocks, num_state_slots, model.device())
    }

    /// The bytes the cache may take, which is what the budget leaves once the weights and the
    /// peak of one forward pass are taken out. Not positive when nothing is left.
    fn estimate_memory_budget<M: ModelForGeneration>(
        model: &M,
        config: &EngineConfig,
        spec: &ModelCacheSpec,
        num_state_slots: i32,
    ) -> Result<i64> {
        if !(0.0..=1.0).contains(&config.kv_cache_memory_utilization)
            || config.kv_cache_memory_utilization <= 0.0
        {
            return Err(Error::model(
                "kv_cache_memory_utilization must be within (0, 1]",
            ));
        }
        if config.max_num_batched_tokens <= 0 {
            return Err(Error::model("max_num_batched_tokens must be positive"));
        }
        if spec.has_recurrent_state() && config.max_num_seqs <= 0 {
            return Err(Error::model(
                "a model with recurrent layers needs a positive max_num_seqs",
            ));
        }

        let device = model.device();
        let block_size = config.kv_cache_block_size;
        let num_tokens = config
            .max_num_batched_tokens
            .min(spec.max_context_length());

        // The pass has to put its keys and its state somewhere, so it is profiled against a scratch
        // cache just large enough for one batch. That cache's size is known exactly and comes back
        // out of the measurement below, leaving the weights and the activation.
        let num_scratch_blocks = (num_tokens + block_size - 1) / block_size;
        let num_scratch_slots = if spec.has_recurrent_state() { 1 } else { 0 };
        let scratch_bytes = spec.bytes_per_block(block_size) * i64::from(num_scratch_blocks)
            + spec.bytes_per_state_slot() * i64::from(num_scratch_slots);
        let peak = {
            let mut scratch = KVCacheManager::new(
                spec.clone(),
                block_size,
                num_scratch_blocks.max(1),
                num_scratch_slots,
                device,
            )?;
            let block_ids = scratch
                .allocate_blocks_for_tokens(num_tokens)
                .ok_or_else(|| Error::model("could not allocate the profiling scratch pool"))?;
            let state_slot = if scratch.needs_state_slot() {
                Some(scratch.allocate_state_slot().ok_or_else(|| {
                    Error::model("could not allocate the profiling state slot")
                })?)
            } else {
                None
            };

            let mut batch = ForwardBatch::single(&vec![0i64; num_tokens as usize], 0)?;
            batch.set_block_ids(vec![block_ids])?;
            if let Some(slot) = state_slot {
                batch.set_state_slots(vec![slot])?;
            }
            let prepared = batch.prepare(device, block_size)?;

            MemorySnapshot::reset_peak_stats(device)?;
            model.forward(&prepared, &mut scratch)?;
            MemorySnapshot::capture(device)?
        };

        if peak.total <= 0 {
            return Err(Error::model(
                "device does not report its memory usage, so the kv cache cannot be sized",
            ));
        }

        let peak_bytes = peak.peak_allocated - scratch_bytes;
        let budget =
            (peak.total as f64 * f64::from(config.kv_cache_memory_utilization)) as i64 - peak_bytes;

        // Another process may hold memory this one never sees. The scratch cache is gone by now,
        // so its bytes are free again even though the snapshot was taken while it was held.
        let budget = budget.min(peak.free + scratch_bytes);

        // A model with recurrent layers cannot run max_num_seqs requests without max_num_seqs
        // slots, so this is worth saying plainly here rather than as "no blocks left" below.
        let state_bytes = spec.bytes_per_state_slot() * i64::from(num_state_slots);
        if state_bytes > 0 && budget <= state_bytes {
            return Err(Error::model(format!(
                "the recurrent state of {num_state_slots} requests needs {state_bytes} bytes and \
                 only {budget} are left; lower max_num_seqs or raise \
                 kv_cache_memory_utilization"
            )));
        }

        Ok(budget)
    }
}
