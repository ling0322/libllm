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

//! Measures how fast a model reads a prompt and how fast it generates.
//!
//! ```text
//! cargo run --release --example benchmark -- models/llama3.2-3b-instruct-fp16.llmpkg [repeats]
//! ```
//!
//! Two numbers, because they are limited by different things: reading a prompt is one large
//! matrix multiplication per layer and is bound by arithmetic, while generating a token at a time
//! reads the whole model for one row and is bound by memory bandwidth.

use std::time::{Duration, Instant};

use llm::flint::Device;
use llm::{EngineConfig, ForwardBatch, KVCacheManager, LlamaForGeneration, Prompt, ZipFile};

/// How long to keep repeating each measurement.
const MAX_WAIT: Duration = Duration::from_secs(5);
const BLOCK_SIZE: i32 = 256;

fn main() -> Result<(), llm::Error> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: benchmark <package.llmpkg> [prompt-repeat-count]");
        std::process::exit(2);
    };
    let repeats: usize = args
        .next()
        .and_then(|count| count.parse().ok())
        .unwrap_or(10);

    let device = if Device::Cuda.is_available() {
        Device::Cuda
    } else {
        eprintln!("this needs a CUDA device");
        std::process::exit(1);
    };

    let package = ZipFile::open(&path)?;
    let model = LlamaForGeneration::from_package(device, &package)?;
    let mut cache = KVCacheManager::for_model(&model, &EngineConfig::default())?;

    let mut prompt = Prompt::new();
    prompt.append_text("The quick brown fox jumps over the lazy dog. ".repeat(repeats));
    let token_ids = model.encode_prompt(&prompt)?;
    let prompt_length = token_ids.len() as i32;

    let prefill = benchmark_prefill(&model, &mut cache, &token_ids)?;
    println!(
        "{:<12} {:<8} prefill@len:{:<5}  {:<7.1} tokens/s",
        model.name(),
        "cuda",
        prompt_length,
        prefill
    );

    let decode = benchmark_decode(&model, &mut cache, &token_ids)?;
    println!(
        "{:<12} {:<8} decode@ctx:{:<5}   {:<7.1} tokens/s",
        model.name(),
        "cuda",
        prompt_length,
        decode
    );

    Ok(())
}

/// Forwards `token_ids` after the `past_len` tokens the blocks already hold. Running the same
/// batch again overwrites the same slots, which is what lets the loops below repeat.
fn forward(
    model: &LlamaForGeneration,
    cache: &mut KVCacheManager,
    block_ids: &[i32],
    token_ids: &[i64],
    past_len: i32,
) -> Result<(), llm::Error> {
    let mut batch = ForwardBatch::single(token_ids, past_len)?;
    batch.set_block_ids(vec![block_ids.to_vec()])?;
    let prepared = batch.prepare(model.device(), BLOCK_SIZE)?;
    model.forward(&prepared, cache)?;
    Ok(())
}

/// Tokens per second while reading a whole prompt at once.
fn benchmark_prefill(
    model: &LlamaForGeneration,
    cache: &mut KVCacheManager,
    token_ids: &[i64],
) -> Result<f64, llm::Error> {
    let block_ids = cache
        .allocate_blocks_for_tokens(token_ids.len() as i32)
        .expect("the cache should hold one prompt");

    // One pass first, so that what is measured is not the first call's setup.
    forward(model, cache, &block_ids, token_ids, 0)?;

    let start = Instant::now();
    let mut loops = 0;
    while start.elapsed() < MAX_WAIT {
        forward(model, cache, &block_ids, token_ids, 0)?;
        loops += 1;
    }
    let elapsed = start.elapsed();

    cache.free_blocks(&block_ids)?;
    Ok(token_ids.len() as f64 * f64::from(loops) / elapsed.as_secs_f64())
}

/// Tokens per second while generating one token at a time on top of that prompt.
fn benchmark_decode(
    model: &LlamaForGeneration,
    cache: &mut KVCacheManager,
    token_ids: &[i64],
) -> Result<f64, llm::Error> {
    let prompt_length = token_ids.len() as i32;
    let block_ids = cache
        .allocate_blocks_for_tokens(prompt_length + 1)
        .expect("the cache should hold one prompt and a token");

    forward(model, cache, &block_ids, token_ids, 0)?;
    forward(model, cache, &block_ids, &[0], prompt_length)?;

    let start = Instant::now();
    let mut loops = 0;
    while start.elapsed() < MAX_WAIT {
        forward(model, cache, &block_ids, &[0], prompt_length)?;
        loops += 1;
    }
    let elapsed = start.elapsed();

    cache.free_blocks(&block_ids)?;
    Ok(f64::from(loops) / elapsed.as_secs_f64())
}
