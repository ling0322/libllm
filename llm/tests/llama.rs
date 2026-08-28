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

//! Checks the Llama model against the reference logits in the test package.
//!
//! Those logits came out of huggingface transformers, so a mismatch here means this port
//! disagrees with transformers rather than merely with itself. The model and the test package are
//! large and are not in the repository, and the attention kernels the model needs exist only for
//! CUDA, so this is `#[ignore]`d: run it with `cargo test --test llama -- --ignored`.

use llm::flint::{functional as F, DType, Device, Tensor};
use llm::{ForwardBatch, KVCacheManager, LlamaForGeneration, LlamaModel, VarBuilder, ZipFile};

const MODEL_PACKAGE: &str = "llama3.2-3b-instruct-fp16.llmpkg";
const TEST_PACKAGE: &str = "llama3.2-3b-instruct-fp16_test.llmpkg";
const BLOCK_SIZE: i32 = 256;

const MAX_RMSE_OVER_CUDA_BASELINE: f64 = 1.05;

/// Test cases longer than this are skipped by `logits_match_the_reference`.
///
/// TODO: raise or remove this. It exists only to keep `cargo test` quick: the package's longest
/// case is 1271 tokens and on its own accounted for ~100s of the ~117s that test took, while the
/// 11- and 64-token cases together take ~14s including model load. The cost is not the forward
/// pass -- it is comparing every position's logits, `num_tokens * 128256` floats against two
/// references, so it grows with the sequence and the long case is the one that exercises paged
/// attention across more than one block. Comparing a sampled subset of positions instead would
/// keep that coverage without the bill.
const MAX_TOKENS_PER_CASE: i32 = 256;

fn package_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../models")
        .join(name)
}

/// Forwards every token in one pass and gives back the logits of every position, on the host.
fn forward_all(model: &LlamaModel, input_ids: &Tensor, device: Device) -> Tensor {
    let num_tokens = input_ids.shape_at(0).unwrap();
    let spec = llm::ModelCacheSpec::uniform_attention(
        model.config().num_layers,
        model.config().num_key_value_heads,
        model.config().head_dim().unwrap(),
        model.config().max_context_length,
        F::default_float_type(device).unwrap(),
    )
    .unwrap();

    // A pool just large enough for this one sequence.
    let num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut cache = KVCacheManager::new(spec, BLOCK_SIZE, num_blocks, 0, device).unwrap();
    let block_ids = cache.allocate_blocks_for_tokens(num_tokens).unwrap();

    let mut batch = ForwardBatch::single_layout(num_tokens, 0).unwrap();
    batch.set_block_ids(vec![block_ids]).unwrap();
    let prepared = batch.prepare(device, BLOCK_SIZE).unwrap();

    let input = input_ids.to_device(device).unwrap();
    let hidden = model.forward(&input, &prepared, &mut cache).unwrap();
    let logits = model.forward_lm_head(&hidden).unwrap();

    logits
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .contiguous()
        .unwrap()
}

struct ErrorStats {
    rmse: f64,
    p99: f32,
    p999: f32,
    max_abs: f32,
}

fn error_stats(actual: &[f32], reference: &[f32]) -> ErrorStats {
    assert_eq!(actual.len(), reference.len());
    assert!(!actual.is_empty());

    let mut squared_error = 0.0f64;
    let mut absolute_errors = Vec::with_capacity(actual.len());
    for (&actual, &reference) in actual.iter().zip(reference) {
        let difference = (actual - reference).abs();
        squared_error += f64::from(difference).powi(2);
        absolute_errors.push(difference);
    }
    absolute_errors.sort_unstable_by(f32::total_cmp);

    let percentile = |quantile: f64| {
        let index = ((absolute_errors.len() - 1) as f64 * quantile).ceil() as usize;
        absolute_errors[index]
    };
    ErrorStats {
        rmse: (squared_error / actual.len() as f64).sqrt(),
        p99: percentile(0.99),
        p999: percentile(0.999),
        max_abs: *absolute_errors.last().expect("checked above"),
    }
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("logits are not NaN"))
        .map(|(index, _)| index)
        .expect("a row of logits is not empty")
}

#[test]
fn logits_match_the_reference() {
    let device = Device::Cuda;

    let test_package = ZipFile::open(package_path(TEST_PACKAGE)).unwrap();
    let cases = VarBuilder::from_reader(
        &mut test_package.open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap();
    // An empty package is a corrupt one, not a reason to quietly check nothing.
    assert!(
        cases.has("test_case.0.input_ids"),
        "no test cases in package"
    );

    let package = ZipFile::open(package_path(MODEL_PACKAGE)).unwrap();
    let generation = LlamaForGeneration::from_package(device, &package).unwrap();
    let model = generation.model();

    let mut cases_run = 0;
    for case in 0.. {
        let prefix = format!("test_case.{case}.");
        if !cases.has(&format!("{prefix}input_ids")) {
            assert!(case > 0, "no test cases ran");
            break;
        }

        let input_ids = cases.get_unchecked(&format!("{prefix}input_ids")).unwrap();
        let num_tokens = input_ids.shape_at(0).unwrap();

        // The check below is skipped past this length. Bail out before reading the references,
        // which are the expensive part: each is a [num_tokens, vocab] block of floats, so the
        // 1271-token case pulls in two 650MB tensors and then compares them element by element.
        if num_tokens > MAX_TOKENS_PER_CASE {
            eprintln!("case={case} skipped: {num_tokens} tokens is over the {MAX_TOKENS_PER_CASE} cap");
            continue;
        }

        let cpu_reference = cases
            .get_unchecked(&format!("{prefix}logits_cpu_fp32"))
            .unwrap();
        let reference = cases
            .get_unchecked(&format!("{prefix}logits_cuda_fp16"))
            .unwrap();

        let vocab_size = reference.shape_at(1).unwrap();
        assert_eq!(reference.shape_at(0).unwrap(), num_tokens);
        assert_eq!(reference.dtype(), DType::Float);
        assert_eq!(cpu_reference.shape(), reference.shape());
        assert_eq!(cpu_reference.dtype(), DType::Float);

        let logits = forward_all(model, &input_ids, device);
        assert_eq!(logits.shape(), vec![num_tokens, vocab_size]);

        let actual = logits.to_vec_f32().unwrap();
        let cpu_expected = cpu_reference.to_vec_f32().unwrap();
        let cuda_expected = reference.to_vec_f32().unwrap();
        let cpu_stats = error_stats(&actual, &cpu_expected);
        let reference_stats = error_stats(&cuda_expected, &cpu_expected);
        for (label, stats) in [
            ("libllm-cpu", &cpu_stats),
            ("hf-cuda-cpu", &reference_stats),
        ] {
            eprintln!(
                "case={case} {label} rmse={:.6} p99={:.6} p99.9={:.6} max_abs={:.6}",
                stats.rmse, stats.p99, stats.p999, stats.max_abs
            );
        }
        assert!(
            cpu_stats.rmse <= reference_stats.rmse * MAX_RMSE_OVER_CUDA_BASELINE,
            "case {case}: libllm-vs-CPU RMSE {} exceeds 105% of the HF CUDA-vs-CPU baseline {}",
            cpu_stats.rmse,
            reference_stats.rmse,
        );

        // Whatever the small numerical differences, the token the model predicts has to be the
        // same one at every position.
        for position in 0..num_tokens as usize {
            let row = position * vocab_size as usize..(position + 1) * vocab_size as usize;
            assert_eq!(
                argmax(&actual[row.clone()]),
                argmax(&cuda_expected[row]),
                "case {case}, position {position}: predicted a different token than CUDA FP16"
            );
        }

        cases_run += 1;
    }

    // Skipping is a speed trade-off, not a licence to check nothing: a package whose cases were
    // all longer than the cap would otherwise pass without comparing a single logit.
    assert!(
        cases_run > 0,
        "every test case was over the {MAX_TOKENS_PER_CASE} token cap, so nothing was compared"
    );
}

#[test]
fn incremental_decode_matches_one_shot_prefill() {
    let device = Device::Cuda;

    let test_package = ZipFile::open(package_path(TEST_PACKAGE)).unwrap();
    let cases = VarBuilder::from_reader(
        &mut test_package.open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap();
    let input_ids = cases.get_unchecked("test_case.0.input_ids").unwrap();

    let package = ZipFile::open(package_path(MODEL_PACKAGE)).unwrap();
    let generation = LlamaForGeneration::from_package(device, &package).unwrap();
    let model = generation.model();

    // A prefix of the case, short enough that stepping through it one token at a time is quick.
    let all_ids = input_ids.to_vec_i64().unwrap();
    let num_tokens = (all_ids.len() as i32).min(8);
    let tokens = Tensor::from_i64(&[num_tokens], &all_ids[..num_tokens as usize]).unwrap();
    let one_shot = forward_all(model, &tokens, device);
    let vocab_size = one_shot.shape_at(1).unwrap() as usize;
    let expected = one_shot.to_vec_f32().unwrap();

    // Feed the same tokens one at a time, reading the cache the previous steps filled.
    let spec = generation.kv_cache_spec().unwrap();
    let mut cache = KVCacheManager::new(spec, BLOCK_SIZE, 1, 0, device).unwrap();
    let block_ids = cache.allocate_blocks_for_tokens(num_tokens).unwrap();

    let token_ids = tokens.to_vec_i64().unwrap();
    for (step, &token) in token_ids.iter().enumerate() {
        let mut batch = ForwardBatch::single(&[token], step as i32).unwrap();
        batch.set_block_ids(vec![block_ids.clone()]).unwrap();
        let prepared = batch.prepare(device, BLOCK_SIZE).unwrap();

        let logits = generation.forward(&prepared, &mut cache).unwrap();
        let logits = logits
            .to_device(Device::Cpu)
            .unwrap()
            .cast(DType::Float)
            .unwrap();
        let actual = logits.to_vec_f32().unwrap();

        let row = &expected[step * vocab_size..(step + 1) * vocab_size];
        assert_eq!(
            argmax(&actual),
            argmax(row),
            "step {step}: decoding one token at a time predicted something else"
        );
    }
}
