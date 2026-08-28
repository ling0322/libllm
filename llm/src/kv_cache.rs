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

use crate::cache_pool::{BlockShape, CachePool, CacheUnit, GroupShape};
use crate::flint::{DType, Device, MemorySnapshot, Tensor};

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

/// The largest number both are a whole multiple of, and whichever is not zero when the other is.
fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
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

    /// The groups the layers fall into, all of them the same size.
    ///
    /// Layers that need the same thing of a block belong together, and layers that agree on
    /// everything but their position are one group however far apart they run. What makes this
    /// more than a `group_by` is the second rule, which vLLM's hybrid allocator has for the same
    /// reason: **every group has the same number of layers**, so a kind that has more of them than
    /// the smallest kind is split into several groups rather than given a larger block. A stack of
    /// 16 full attention layers and 48 gated DeltaNet ones is four groups of 16, not one of 16 and
    /// one of 48.
    ///
    /// That is what lets one pool serve all of them: the pool holds a tensor per position within a
    /// group, and layer `i` of every group reads its blocks out of the same one. A group larger
    /// than another would leave its extra layers with no storage to share.
    ///
    /// The layers of a kind are dealt out to its groups in turn -- the 48 recurrent layers of that
    /// stack go 0, 3, 6, ... to the first and 1, 4, 7, ... to the second -- so a group is a set of
    /// layers spread through the model rather than a slice of it.
    pub fn groups(&self) -> Vec<KVCacheGroup> {
        let mut kinds: Vec<(KVCacheSpec, Vec<i32>)> = Vec::new();
        for (layer, spec) in self.layers.iter().enumerate() {
            match kinds.iter_mut().find(|(kind, _)| kind == spec) {
                Some((_, layers)) => layers.push(layer as i32),
                None => kinds.push((spec.clone(), vec![layer as i32])),
            }
        }

        // The largest group size every kind can be cut into a whole number of.
        let size = kinds
            .iter()
            .map(|(_, layers)| layers.len() as i32)
            .fold(0, gcd);

        let mut groups = Vec::new();
        for (spec, layers) in kinds {
            let count = layers.len() as i32 / size;
            for group in 0..count {
                groups.push(KVCacheGroup {
                    layers: layers
                        .iter()
                        .skip(group as usize)
                        .step_by(count as usize)
                        .copied()
                        .collect(),
                    spec: spec.clone(),
                });
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

    /// What each group needs of one allocation: the tensors a layer of it keeps there, and whether
    /// one allocation is a block or a whole run.
    ///
    /// An attention layer keeps the keys and the values of `block_size` tokens, and takes another
    /// block whenever its sequence outgrows the last one. A recurrent layer keeps the states the
    /// spec lists, and takes one run at the start and holds it.
    fn group_shapes(groups: &[KVCacheGroup], block_size: i32) -> Result<Vec<GroupShape>> {
        groups
            .iter()
            .map(|group| match &group.spec {
                KVCacheSpec::FullAttention(attention) => {
                    let shape = vec![
                        block_size,
                        attention.num_key_value_heads,
                        attention.head_dim,
                    ];
                    GroupShape::new(
                        vec![shape.clone(), shape],
                        attention.dtype,
                        CacheUnit::Block,
                    )
                }
                KVCacheSpec::RecurrentState(recurrent) => {
                    let dtype = recurrent.dtypes[0];
                    if recurrent
                        .dtypes
                        .iter()
                        .any(|&state_dtype| state_dtype != dtype)
                    {
                        return Err(Error::model(
                            "all states in one cache page must have the same type",
                        ));
                    }
                    GroupShape::new(recurrent.shapes.clone(), dtype, CacheUnit::Run)
                }
            })
            .collect()
    }

    /// The three numbers that decide how large a pool for this model is: what one block id costs
    /// across every slot of it, how many blocks one recurrent state spans, and how many states a
    /// request holds.
    ///
    /// Worked out without allocating anything, which is what lets a caller turn a budget in bytes
    /// into a block count before it builds the pool.
    pub fn pool_layout(&self, block_size: i32) -> Result<(i64, i32, i32)> {
        let groups = self.groups();
        let shapes = ModelCacheSpec::group_shapes(&groups, block_size)?;
        let (page_size_bytes, blocks_per_run) = CachePool::page_layout(&shapes)?;

        let num_layers = groups[0].layers.len() as i64;
        let num_state_groups = groups
            .iter()
            .filter(|group| group.spec.kind() == CacheKind::RecurrentState)
            .count() as i32;

        let bytes_per_block = page_size_bytes * num_layers;
        Ok((bytes_per_block, blocks_per_run, num_state_groups))
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
/// Owns the cache storage and hands out blocks and state slots of it.
///
/// One [`CachePool`] holds the whole of it. What a layer of the model reads its blocks through is
/// a view of that pool, decided by the group the layer is in; what a request holds is a list of
/// block ids for the attention group and one run per recurrent group, all drawn from the one
/// allocator.
#[derive(Debug)]
pub struct KVCacheManager {
    spec: ModelCacheSpec,
    block_size: i32,
    pool: CachePool,
    /// Where each model layer sits: its group, and its slot within that group.
    placement: Vec<(i32, i32)>,
    /// The group whose blocks a sequence's block table names. `None` on a model with no attention
    /// layer at all.
    attention_group: Option<i32>,
    /// The groups that take a run per request, in the order a request's slots are given in. Empty
    /// on a model that keeps no state between tokens.
    state_groups: Vec<i32>,
}

impl KVCacheManager {
    /// Allocate the pool every layer draws from.
    ///
    /// `block_size` is a power of two, which is what lets a token's position be split into a block
    /// and an offset without a division at every step. `num_blocks` is how large the pool is, in
    /// blocks; it is rounded down to a whole number of runs. `num_state_slots` is how many requests
    /// may hold a recurrent state at once, and so how many may run at once on a model that has one
    /// -- their runs are kept back from the blocks so that a request that has been admitted can
    /// always be given its state. It is ignored by a model that has no recurrent layer.
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
        if num_blocks <= 0 {
            return Err(Error::model(format!(
                "a kv cache needs a positive block count, got {num_blocks}"
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

        let groups = spec.groups();
        let shapes = ModelCacheSpec::group_shapes(&groups, block_size)?;

        let mut placement = vec![(0, 0); spec.layers.len()];
        let mut attention_group = None;
        let mut state_groups = Vec::new();
        for (index, group) in groups.iter().enumerate() {
            let index = index as i32;
            for (slot, &layer) in group.layers.iter().enumerate() {
                placement[layer as usize] = (index, slot as i32);
            }
            match group.spec.kind() {
                CacheKind::FullAttention => {
                    if attention_group.is_some() {
                        // Every attention layer reads the blocks of one table, so a model whose
                        // attention layers fall into more than one group would need a table per
                        // group to run. Nothing builds one yet.
                        return Err(Error::model(
                            "this cache runs a model whose attention layers are all one group, \
                             and these are not",
                        ));
                    }
                    attention_group = Some(index);
                }
                CacheKind::RecurrentState => state_groups.push(index),
            }
        }

        // A request that has been admitted holds one run of every recurrent group, and it cannot
        // run without them, so that many runs are what the blocks may not break into.
        let reserved_runs = num_state_slots * state_groups.len() as i32;
        let (page_size_bytes, _) = CachePool::page_layout(&shapes)?;
        let shape = BlockShape::new(page_size_bytes, groups[0].layers.len() as i32, num_blocks)?;
        let pool = CachePool::new(shape, shapes, reserved_runs, device)?;

        Ok(KVCacheManager {
            spec,
            block_size,
            pool,
            placement,
            attention_group,
            state_groups,
        })
    }

    /// The key pool of one attention layer, `<dtype>(numBlocks, blockSize, numKeyValueHeads,
    /// headDim)`.
    pub fn key_cache(&self, layer: i32) -> Result<&Tensor> {
        Ok(&self.attention_layer(layer)?[0])
    }

    /// The value pool of one attention layer, shaped like the key pool.
    pub fn value_cache(&self, layer: i32) -> Result<&Tensor> {
        Ok(&self.attention_layer(layer)?[1])
    }

    /// The states of one recurrent layer, each `<dtype>(numSlots, ...)` and in the order the spec
    /// gave them. A gated DeltaNet layer has the recurrence's state first and the convolution's
    /// window second.
    ///
    /// They are views of the pool rather than pools of their own, so they are not contiguous: a
    /// slot is a run of the pool and the state is one region of it.
    pub fn state_cache(&self, layer: i32) -> Result<&[Tensor]> {
        let (group, slot) = self.place(layer, CacheKind::RecurrentState)?;
        self.pool.layer(group, slot)
    }

    /// The states of one recurrent layer, to write into.
    pub fn state_cache_mut(&mut self, layer: i32) -> Result<&mut [Tensor]> {
        let (group, slot) = self.place(layer, CacheKind::RecurrentState)?;
        self.pool.layer_mut(group, slot)
    }

    /// Where a layer sits in the pool, and an error when it is of the other kind.
    fn place(&self, layer: i32, kind: CacheKind) -> Result<(i32, i32)> {
        if self.spec.layer(layer)?.kind() != kind {
            return Err(Error::model(format!(
                "layer {layer} is not a {} layer",
                match kind {
                    CacheKind::FullAttention => "full attention",
                    CacheKind::RecurrentState => "recurrent",
                }
            )));
        }
        Ok(self.placement[layer as usize])
    }

    fn attention_layer(&self, layer: i32) -> Result<&[Tensor]> {
        let (group, slot) = self.place(layer, CacheKind::FullAttention)?;
        self.pool.layer(group, slot)
    }

    /// The key and value pools of one layer, to write into.
    ///
    /// Both at once, because a forward pass writes the keys and the values of a layer together and
    /// two separate borrows of the same manager would not be allowed.
    pub fn caches_mut(&mut self, layer: i32) -> Result<(&mut Tensor, &mut Tensor)> {
        let (group, slot) = self.place(layer, CacheKind::FullAttention)?;
        let tensors = self.pool.layer_mut(group, slot)?;

        let (keys, values) = tensors.split_at_mut(1);
        Ok((&mut keys[0], &mut values[0]))
    }

    pub fn spec(&self) -> &ModelCacheSpec {
        &self.spec
    }

    /// The tokens one block holds.
    pub fn block_size(&self) -> i32 {
        self.block_size
    }

    /// The pool the whole model draws from.
    pub fn pool(&self) -> &CachePool {
        &self.pool
    }

    pub fn num_blocks(&self) -> i32 {
        self.pool.num_blocks()
    }

    pub fn num_free_blocks(&self) -> i32 {
        self.pool.num_free_blocks()
    }

    /// How many requests may hold a recurrent state at once. Zero for a model without one.
    pub fn num_state_slots(&self) -> i32 {
        match self.state_groups.len() as i32 {
            0 => 0,
            groups => self.pool.num_runs() / groups,
        }
    }

    /// How many more requests could be handed one.
    pub fn num_free_state_slots(&self) -> i32 {
        match self.state_groups.len() as i32 {
            0 => 0,
            groups => self.pool.num_free_runs() / groups,
        }
    }

    /// Whether a request has to hold a state slot to run at all.
    pub fn needs_state_slot(&self) -> bool {
        !self.state_groups.is_empty()
    }

    /// Which of a request's state slots a recurrent layer reads: the layers of one group all read
    /// the same one, and a request holds one per group.
    pub fn state_group_of(&self, layer: i32) -> Result<i32> {
        let (group, _) = self.place(layer, CacheKind::RecurrentState)?;
        self.state_groups
            .iter()
            .position(|&candidate| candidate == group)
            .map(|index| index as i32)
            .ok_or_else(|| Error::model(format!("layer {layer} is not in a recurrent group")))
    }

    /// How many state slots a request holds, which is one per recurrent group.
    pub fn num_state_groups(&self) -> i32 {
        self.state_groups.len() as i32
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

    /// Take `num_blocks` blocks, in ascending order.
    ///
    /// All or nothing: a sequence that gets half of what it asked for could not run anyway, and
    /// leaving the pool untouched keeps a scheduler from having to hand blocks back.
    pub fn allocate_blocks(&mut self, num_blocks: i32) -> Option<Vec<i32>> {
        let mut blocks = Vec::with_capacity(num_blocks.max(0) as usize);
        if self.allocate_blocks_into(num_blocks, &mut blocks) {
            Some(blocks)
        } else {
            None
        }
    }

    /// The same, appending to a buffer the caller owns. A scheduler that grows a sequence every
    /// step can keep one buffer and allocate nothing to do it.
    pub fn allocate_blocks_into(&mut self, num_blocks: i32, out: &mut Vec<i32>) -> bool {
        if self.attention_group.is_none() {
            return num_blocks == 0;
        }
        self.pool.allocate_blocks(num_blocks, out)
    }

    /// Take the blocks `num_tokens` tokens need.
    pub fn allocate_blocks_for_tokens(&mut self, num_tokens: i32) -> Option<Vec<i32>> {
        self.allocate_blocks(self.num_blocks_for_tokens(num_tokens))
    }

    /// The blocks a sequence may still be given, which is what is free less what is kept back for
    /// the states of the requests that have not been admitted yet.
    pub fn num_allocatable_blocks(&self) -> i32 {
        self.pool.num_allocatable_blocks()
    }

    /// Give blocks back.
    pub fn free_blocks(&mut self, block_ids: &[i32]) -> Result<()> {
        if self.attention_group.is_none() && !block_ids.is_empty() {
            return Err(Error::model(
                "this model has no attention layer to free blocks of",
            ));
        }
        self.pool.free_blocks(block_ids)
    }

    /// Take a state slot of every recurrent group, or nothing when they cannot all be had.
    ///
    /// A slot belongs to one request for as long as it runs: what it holds is that request's whole
    /// history folded together, so unlike a block it cannot be handed to somebody else and read
    /// back later. The slots come back zeroed, which is the state a sequence starts from.
    pub fn allocate_state_slots(&mut self) -> Option<Vec<i32>> {
        if self.state_groups.is_empty() {
            return Some(Vec::new());
        }

        let mut slots = Vec::with_capacity(self.state_groups.len());
        for _ in 0..self.state_groups.len() {
            match self.pool.allocate_run() {
                // Whatever the request before it left here is not a prefix of this one's history.
                // Zeroing on the way out rather than on the way in keeps a slot that is never
                // reused from being cleared twice, and there is nothing to read between the two.
                Some(run) if self.pool.clear_run(run).is_ok() => slots.push(run),
                Some(run) => {
                    let _ = self.pool.free_run(run);
                    let _ = self.free_state_slots(&slots);
                    return None;
                }
                None => {
                    // A request needs a slot in every group or none: one it cannot fill would stop
                    // it running, and holding the others would keep them from a request that could.
                    let _ = self.free_state_slots(&slots);
                    return None;
                }
            }
        }
        Some(slots)
    }

    /// Give a request's state slots back.
    pub fn free_state_slots(&mut self, slots: &[i32]) -> Result<()> {
        if self.state_groups.is_empty() && !slots.is_empty() {
            return Err(Error::model(
                "this model keeps no state between tokens, so it has no slots to free",
            ));
        }
        for &slot in slots {
            self.pool.free_run(slot)?;
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
    /// What is left is spent on blocks, and the states come out of the same blocks: a request's
    /// runs are kept back from the block allocator rather than held in a pool of their own, so a
    /// model that is running fewer requests than `max_num_seqs` spends the difference on context.
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

        let (bytes_per_block, blocks_per_run, num_state_groups) =
            spec.pool_layout(block_size)?;
        let budget = Self::estimate_memory_budget(model, config, &spec, num_state_slots)?;

        let num_blocks = (budget / bytes_per_block).min(i64::from(i32::MAX)) as i32;
        let num_runs = num_blocks / blocks_per_run;
        if num_runs <= 0 {
            return Err(Error::model(
                "not enough device memory left for even one run of the kv cache",
            ));
        }

        // A model with recurrent layers cannot run max_num_seqs requests without a run per group
        // for each of them, and the ones left over are what its context is held in.
        let reserved_runs = i64::from(num_state_slots) * i64::from(num_state_groups);
        if i64::from(num_runs) <= reserved_runs {
            return Err(Error::model(format!(
                "the recurrent state of {num_state_slots} requests needs {reserved_runs} runs of \
                 the kv cache and only {num_runs} fit; lower max_num_seqs or raise \
                 kv_cache_memory_utilization"
            )));
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
        let (bytes_per_block, blocks_per_run, num_state_groups) = spec.pool_layout(block_size)?;

        // The pass has to put its keys and its state somewhere, so it is profiled against a scratch
        // cache just large enough for one batch and one request's state. That cache's size is known
        // exactly and comes back out of the measurement below, leaving the weights and the
        // activation.
        let num_scratch_blocks = (num_tokens + block_size - 1) / block_size
            + if num_state_slots > 0 {
                blocks_per_run * num_state_groups
            } else {
                0
            };
        let scratch_bytes = bytes_per_block * i64::from(num_scratch_blocks);
        let peak = {
            let mut scratch = KVCacheManager::new(
                spec.clone(),
                block_size,
                num_scratch_blocks.max(blocks_per_run),
                if num_state_slots > 0 { 1 } else { 0 },
                device,
            )?;
            let state_slots = scratch.allocate_state_slots().ok_or_else(|| {
                Error::model("could not allocate the profiling state slots")
            })?;
            let block_ids = scratch
                .allocate_blocks_for_tokens(num_tokens)
                .ok_or_else(|| Error::model("could not allocate the profiling scratch pool"))?;

            let mut batch = ForwardBatch::single(&vec![0i64; num_tokens as usize], 0)?;
            batch.set_block_ids(vec![block_ids])?;
            if !state_slots.is_empty() {
                batch.set_state_slots(
                    state_slots.iter().map(|&slot| vec![slot]).collect::<Vec<_>>(),
                )?;
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
        if budget < bytes_per_block {
            return Err(Error::model(
                "not enough device memory left for even one block of the kv cache",
            ));
        }

        Ok(budget)
    }
}
