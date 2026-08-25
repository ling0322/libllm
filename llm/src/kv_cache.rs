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

//! The paged KV cache: one pool of blocks per layer, handed out to sequences a block at a time.
//!
//! A block holds the keys and values of a fixed number of consecutive tokens. A block id names the
//! same token range in every layer, but each layer addresses it in its own key and value tensor,
//! so a sequence owns a list of block ids and every layer reads its own pool through that list.

use crate::flint::{DType, Device, MemorySnapshot, Tensor};

use crate::engine_config::EngineConfig;
use crate::error::{Error, Result};
use crate::forward_batch::ForwardBatch;
use crate::model::ModelForGeneration;

/// The layout of the cache a model needs, which follows from the shape of its attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KVCacheSpec {
    /// The attention layers that produce cache entries.
    pub num_layers: i32,
    /// The key-value heads of each layer, which is fewer than the query heads under grouped-query
    /// attention.
    pub num_key_value_heads: i32,
    /// The width of one head.
    pub head_dim: i32,
    /// The most tokens the model can attend over, which bounds what one sequence can own.
    pub max_context_length: i32,
    /// The type the keys and values are stored in.
    pub dtype: DType,
}

impl KVCacheSpec {
    /// The bytes one block takes, counting the key and the value of every layer.
    pub fn bytes_per_block(&self, block_size: i32) -> i64 {
        let elements = 2
            * self.num_layers as i64
            * block_size as i64
            * self.num_key_value_heads as i64
            * self.head_dim as i64;
        self.dtype.total_size(elements)
    }
}

/// Owns the cache storage and hands out blocks of it.
#[derive(Debug)]
pub struct KVCacheManager {
    spec: KVCacheSpec,
    block_size: i32,
    num_blocks: i32,
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    /// The blocks nobody holds, kept in descending order so that the lowest id is taken first.
    free_blocks: Vec<i32>,
}

impl KVCacheManager {
    /// Allocate the pools of every layer.
    ///
    /// `block_size` is a power of two, which is what lets a token's position be split into a block
    /// and an offset without a division at every step.
    pub fn new(
        spec: KVCacheSpec,
        block_size: i32,
        num_blocks: i32,
        device: Device,
    ) -> Result<KVCacheManager> {
        if block_size <= 0 || num_blocks <= 0 {
            return Err(Error::model(format!(
                "kv cache needs a positive block size and block count, got {block_size} and \
                 {num_blocks}"
            )));
        }
        if block_size & (block_size - 1) != 0 {
            return Err(Error::model(format!(
                "kv cache block size must be a power of two, got {block_size}"
            )));
        }

        let shape = [
            num_blocks,
            block_size,
            spec.num_key_value_heads,
            spec.head_dim,
        ];
        let mut key_cache = Vec::with_capacity(spec.num_layers as usize);
        let mut value_cache = Vec::with_capacity(spec.num_layers as usize);
        for _ in 0..spec.num_layers {
            // Every element is written by store_kv_cache() before attention reads it, so there is
            // nothing to gain from zeroing what can be tens of gigabytes.
            key_cache.push(Tensor::empty(&shape, spec.dtype, device)?);
            value_cache.push(Tensor::empty(&shape, spec.dtype, device)?);
        }

        Ok(KVCacheManager {
            spec,
            block_size,
            num_blocks,
            key_cache,
            value_cache,
            free_blocks: (0..num_blocks).rev().collect(),
        })
    }

    /// The key pool of one layer, `<dtype>(numBlocks, blockSize, numKeyValueHeads, headDim)`.
    pub fn key_cache(&self, layer: i32) -> Result<&Tensor> {
        self.key_cache
            .get(layer as usize)
            .ok_or_else(|| Error::model(format!("layer {layer} is not in the kv cache")))
    }

    /// The value pool of one layer, shaped like the key pool.
    pub fn value_cache(&self, layer: i32) -> Result<&Tensor> {
        self.value_cache
            .get(layer as usize)
            .ok_or_else(|| Error::model(format!("layer {layer} is not in the kv cache")))
    }

    /// Build the cache a model should have, filling the device memory that is left once the
    /// weights and one full-size forward pass are accounted for.
    ///
    /// How much a pass takes is measured rather than guessed: a full-size batch is forwarded
    /// against a scratch pool just large enough to hold it, and the peak that follows is what the
    /// cache has to leave room for. Needs a device that reports its memory usage, which the CPU
    /// backend does not.
    pub fn for_model<M: ModelForGeneration>(
        model: &M,
        config: &EngineConfig,
    ) -> Result<KVCacheManager> {
        let spec = model.kv_cache_spec()?;
        let block_size = config.kv_cache_block_size;
        let bytes_per_block = spec.bytes_per_block(block_size);
        let budget = Self::estimate_memory_budget(model, config, &spec)?;

        let num_blocks = if budget > 0 {
            budget / bytes_per_block
        } else {
            0
        };
        if num_blocks <= 0 {
            return Err(Error::model(
                "not enough device memory left for even one block of the kv cache",
            ));
        }

        KVCacheManager::new(
            spec,
            block_size,
            num_blocks.min(i64::from(i32::MAX)) as i32,
            model.device(),
        )
    }

    /// The bytes the cache may take, which is what the budget leaves once the weights and the
    /// peak of one forward pass are taken out. Not positive when nothing is left.
    fn estimate_memory_budget<M: ModelForGeneration>(
        model: &M,
        config: &EngineConfig,
        spec: &KVCacheSpec,
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

        let device = model.device();
        let block_size = config.kv_cache_block_size;
        let num_tokens = config.max_num_batched_tokens.min(spec.max_context_length);

        // The pass has to put its keys somewhere, so it is profiled against a scratch pool just
        // large enough for one batch. That pool's size is known exactly and comes back out of the
        // measurement below, leaving the weights and the activation.
        let num_scratch_blocks = (num_tokens + block_size - 1) / block_size;
        let scratch_bytes = spec.bytes_per_block(block_size) * i64::from(num_scratch_blocks);
        let peak = {
            let mut scratch = KVCacheManager::new(*spec, block_size, num_scratch_blocks, device)?;
            let block_ids = scratch
                .allocate_blocks_for_tokens(num_tokens)
                .ok_or_else(|| Error::model("could not allocate the profiling scratch pool"))?;

            let mut batch = ForwardBatch::single(&vec![0i64; num_tokens as usize], 0)?;
            batch.set_block_ids(vec![block_ids])?;
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

        // Another process may hold memory this one never sees. The scratch pool is gone by now,
        // so its bytes are free again even though the snapshot was taken while it was held.
        Ok(budget.min(peak.free + scratch_bytes))
    }

    /// The key and value pools of one layer, to write into.
    ///
    /// Both at once, because a forward pass writes the keys and the values of a layer together and
    /// two separate borrows of the same manager would not be allowed.
    pub fn caches_mut(&mut self, layer: i32) -> Result<(&mut Tensor, &mut Tensor)> {
        let index = layer as usize;
        if index >= self.key_cache.len() {
            return Err(Error::model(format!(
                "layer {layer} is not in the kv cache"
            )));
        }
        Ok((&mut self.key_cache[index], &mut self.value_cache[index]))
    }

    pub fn spec(&self) -> &KVCacheSpec {
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

    /// The blocks `num_tokens` tokens need.
    pub fn num_blocks_for_tokens(&self, num_tokens: i32) -> i32 {
        (num_tokens + self.block_size - 1) / self.block_size
    }

    /// The blocks a sequence that runs to the end of the context would need, which is what a
    /// scheduler reserves room for.
    pub fn max_num_blocks_per_request(&self) -> i32 {
        self.num_blocks_for_tokens(self.spec.max_context_length)
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
}
