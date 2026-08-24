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

use flint::{functional as F, DType, Device, Tensor};
use llm::{ForwardBatch, KVCacheManager, LlamaForGeneration, LlamaModel, VarBuilder, ZipFile};

const MODEL_PACKAGE: &str = "llama3.2-3b-instruct-fp16.llmpkg";
const TEST_PACKAGE: &str = "llama3.2-3b-instruct-fp16_test.llmpkg";
const BLOCK_SIZE: i32 = 256;

/// The largest relative difference from the reference that still counts as agreement, matching
/// what the C++ test allows for.
const MAX_REL_DIFF: f64 = 0.02;
const MAX_LONG_CONTEXT_REL_DIFF: f64 = 0.04;

fn package_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(name)
}

/// Forwards every token in one pass and gives back the logits of every position, on the host.
fn forward_all(model: &LlamaModel, input_ids: &Tensor, device: Device) -> Tensor {
    let num_tokens = input_ids.shape_at(0).unwrap();
    let spec = llm::KVCacheSpec {
        num_layers: model.config().num_layers,
        num_key_value_heads: model.config().num_key_value_heads,
        head_dim: model.config().head_dim().unwrap(),
        max_context_length: model.config().max_context_length,
        dtype: F::default_float_type(device).unwrap(),
    };

    // A pool just large enough for this one sequence.
    let num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut cache = KVCacheManager::new(spec, BLOCK_SIZE, num_blocks, device).unwrap();
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

/// The distance from the reference, by the same measure the C++ test uses: the largest absolute
/// difference over the mean magnitude of the reference.
fn rel_diff(actual: &[f32], reference: &[f32]) -> f64 {
    let max_diff = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let mean_abs =
        reference.iter().map(|x| (*x as f64).abs()).sum::<f64>() / reference.len() as f64;

    max_diff / mean_abs
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

    for case in 0.. {
        let prefix = format!("test_case.{case}.");
        if !cases.has(&format!("{prefix}input_ids")) {
            assert!(case > 0, "no test cases ran");
            break;
        }

        let input_ids = cases.get_unchecked(&format!("{prefix}input_ids")).unwrap();
        let reference = cases.get_unchecked(&format!("{prefix}logits")).unwrap();

        let num_tokens = input_ids.shape_at(0).unwrap();
        let vocab_size = reference.shape_at(1).unwrap();
        assert_eq!(reference.shape_at(0).unwrap(), num_tokens);

        let logits = forward_all(model, &input_ids, device);
        assert_eq!(logits.shape(), vec![num_tokens, vocab_size]);

        let actual = logits.to_vec_f32().unwrap();
        let expected = reference.to_vec_f32().unwrap();
        let diff = rel_diff(&actual, &expected);
        assert!(
            diff < MAX_REL_DIFF,
            "case {case}: relative difference {diff}"
        );

        // Whatever the small numerical differences, the token the model predicts has to be the
        // same one at every position.
        for position in 0..num_tokens as usize {
            let row = position * vocab_size as usize..(position + 1) * vocab_size as usize;
            assert_eq!(
                argmax(&actual[row.clone()]),
                argmax(&expected[row]),
                "case {case}, position {position}: predicted a different token"
            );
        }
    }
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

    // Cross a cache-block boundary one token at a time, growing the block table as the scheduler
    // does. This also catches cache corruption that accumulates over many decode steps.
    let case_ids = input_ids.to_vec_i64().unwrap();
    let num_tokens = BLOCK_SIZE + 2;
    let all_ids: Vec<i64> = case_ids
        .iter()
        .copied()
        .cycle()
        .take(num_tokens as usize)
        .collect();
    let tokens = Tensor::from_i64(&[num_tokens], &all_ids).unwrap();
    let one_shot = forward_all(model, &tokens, device);
    let vocab_size = one_shot.shape_at(1).unwrap() as usize;
    let expected = one_shot.to_vec_f32().unwrap();

    let spec = generation.kv_cache_spec().unwrap();
    let mut cache = KVCacheManager::new(spec, BLOCK_SIZE, 2, device).unwrap();
    let mut block_ids = cache.allocate_blocks(1).unwrap();

    let token_ids = tokens.to_vec_i64().unwrap();
    for step in 0..num_tokens {
        if step == BLOCK_SIZE {
            block_ids.extend(cache.allocate_blocks(1).unwrap());
        }
        let mut batch = ForwardBatch::single(&[token_ids[step as usize]], step).unwrap();
        batch.set_block_ids(vec![block_ids.clone()]).unwrap();
        let prepared = batch.prepare(device, BLOCK_SIZE).unwrap();

        let logits = generation.forward(&prepared, &mut cache).unwrap();
        let logits = logits
            .to_device(Device::Cpu)
            .unwrap()
            .cast(DType::Float)
            .unwrap();
        let actual = logits.to_vec_f32().unwrap();

        let row = &expected[step as usize * vocab_size..(step as usize + 1) * vocab_size];
        let actual_argmax = argmax(&actual);
        let expected_argmax = argmax(row);
        let diff = rel_diff(&actual, row);
        if step % 16 == 0 || actual_argmax != expected_argmax {
            eprintln!(
                "step={step} rel_diff={diff:.5} actual_argmax={actual_argmax} \
                 actual_max={:.5} actual_at_expected={:.5} expected_argmax={expected_argmax} \
                 expected_max={:.5} expected_at_actual={:.5}",
                actual[actual_argmax],
                actual[expected_argmax],
                row[expected_argmax],
                row[actual_argmax]
            );
        }
        assert!(
            diff < MAX_LONG_CONTEXT_REL_DIFF,
            "step {step}: relative difference {diff}"
        );
    }
}
