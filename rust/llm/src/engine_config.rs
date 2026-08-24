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

//! How much the engine may use, and in what units.

/// Top-level configuration for the engine.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    /// The most tokens one forward pass may take. Also the batch size the peak activation is
    /// profiled at, so raising it costs cache capacity as well as buying throughput.
    pub max_num_batched_tokens: i32,
    /// The tokens one kv cache block holds. A power of two, and the vendored FlashAttention
    /// wants at least 256.
    pub kv_cache_block_size: i32,
    /// The share of the device's memory the engine may use, in (0, 1]. The cache gets what is
    /// left of it once the weights and the peak activation are accounted for.
    pub kv_cache_memory_utilization: f32,
}

impl Default for EngineConfig {
    fn default() -> EngineConfig {
        EngineConfig {
            max_num_batched_tokens: 2048,
            kv_cache_block_size: 256,
            kv_cache_memory_utilization: 0.9,
        }
    }
}
