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

//! Tests for the scheduler and the engine, driven by a stand-in model.
//!
//! The scheduling decisions — chunking a long prompt, batching several requests, preempting one
//! to make room for another, stopping — are what these check, and none of them depend on what a
//! real model computes. A stand-in model that returns logits of the caller's choosing keeps all
//! of it on the CPU, where the attention kernels a real model needs do not exist.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use llm::flint::{Device, Tensor};
use llm::{
    EngineConfig, FinishReason, GenerationConfig, KVCacheManager, KVCacheSpec, ModelForGeneration,
    Prompt, Request, RequestOutput, Scheduler, Tokenizer,
};

/// The vocabulary the stand-in model speaks: a handful of one-character tokens.
const VOCAB: [&str; 8] = ["<unk>", " ", "a", "b", "c", "d", "e", "<eot>"];
const STOP_TOKEN: i32 = 7;

/// Writes a vocabulary in the format the tokenizer reads, so that the stand-in model can carry a
/// real one rather than a second implementation of the same thing.
fn write_vocabulary() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LLsp");
    out.extend_from_slice(&(VOCAB.len() as i32).to_le_bytes());
    out.extend_from_slice(&0x55aai16.to_le_bytes());

    for (id, piece) in VOCAB.iter().enumerate() {
        // The unknown token and the stop token are control tokens; the rest are ordinary text.
        let flag: i8 = match id {
            0 => 1,
            7 => 2,
            _ => 0,
        };
        out.push(flag as u8);

        let stored = if flag == 0 { *piece } else { "" };
        out.push(stored.len() as u8);
        out.extend_from_slice(stored.as_bytes());
        out.push(piece.len() as u8);
        out.extend_from_slice(piece.as_bytes());
        out.extend_from_slice(&(-(id as f32)).to_le_bytes());
    }

    out.extend_from_slice(&0x55aai16.to_le_bytes());
    out
}

/// A package holding nothing but that vocabulary, which is all a tokenizer needs.
fn write_package(path: &std::path::Path) {
    let vocabulary = write_vocabulary();
    let config = b"[tokenizer]\ntype = bpe\nmodel_file = tokenizer.bin\nadd_prefix_space = false\nsplit_by_unicode = true\n";

    let mut file = std::fs::File::create(path).unwrap();
    let mut offset = 0u32;
    let mut directory = Vec::new();

    for (name, data) in [
        ("tokenizer.bin", vocabulary.as_slice()),
        ("tokenizer.ini", config.as_slice()),
    ] {
        let mut header = Vec::new();
        header.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        header.extend_from_slice(&[10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());
        header.extend_from_slice(&(name.len() as u16).to_le_bytes());
        header.extend_from_slice(&[0, 0]);
        header.extend_from_slice(name.as_bytes());

        let local_offset = offset;
        file.write_all(&header).unwrap();
        file.write_all(data).unwrap();
        offset += (header.len() + data.len()) as u32;

        directory.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        directory.extend_from_slice(&[10, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
        directory.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        directory.extend_from_slice(&local_offset.to_le_bytes());
        directory.extend_from_slice(name.as_bytes());
    }

    let directory_offset = offset;
    file.write_all(&directory).unwrap();

    let mut end = Vec::new();
    end.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    end.extend_from_slice(&[0, 0, 0, 0, 2, 0, 2, 0]);
    end.extend_from_slice(&(directory.len() as u32).to_le_bytes());
    end.extend_from_slice(&directory_offset.to_le_bytes());
    end.extend_from_slice(&[0, 0]);
    file.write_all(&end).unwrap();
}

fn tokenizer() -> Tokenizer {
    let dir = std::env::temp_dir().join(format!(
        "llm-serving-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mock.llmpkg");
    write_package(&path);

    let package = llm::ZipFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_package(&package).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    tokenizer
}

/// A model that computes nothing and says what it was told to say.
struct MockModel {
    tokenizer: Tokenizer,
    /// The token to produce next, one per forward, repeating the last one once they run out.
    tokens: Mutex<Vec<i64>>,
    /// How many forward passes it has been asked for, and how big each one was.
    passes: Mutex<Vec<i32>>,
    forwards: AtomicUsize,
    /// Fails every forward, for testing what a broken pass does to the requests in it.
    fails: bool,
}

impl MockModel {
    fn new(tokens: Vec<i64>) -> MockModel {
        MockModel {
            tokenizer: tokenizer(),
            tokens: Mutex::new(tokens),
            passes: Mutex::new(Vec::new()),
            forwards: AtomicUsize::new(0),
            fails: false,
        }
    }

    fn failing() -> MockModel {
        MockModel {
            fails: true,
            ..MockModel::new(vec![2])
        }
    }

    fn spec() -> KVCacheSpec {
        KVCacheSpec {
            num_layers: 1,
            num_key_value_heads: 1,
            head_dim: 8,
            max_context_length: 64,
            dtype: llm::flint::DType::Float,
        }
    }
}

impl ModelForGeneration for MockModel {
    fn forward(
        &self,
        batch: &llm::PreparedBatch,
        _cache: &mut KVCacheManager,
    ) -> llm::Result<Tensor> {
        if self.fails {
            return Err(llm::Error::Model("the model broke".to_string()));
        }

        let pass = self.forwards.fetch_add(1, Ordering::SeqCst);
        self.passes.lock().unwrap().push(batch.total_q_len());

        let tokens = self.tokens.lock().unwrap();
        let token = *tokens.get(pass).unwrap_or_else(|| tokens.last().unwrap());

        // One row per sequence, each favouring the token this pass was told to produce.
        let vocab_size = VOCAB.len();
        let mut logits = vec![0.0f32; batch.num_sequences() as usize * vocab_size];
        for row in 0..batch.num_sequences() as usize {
            logits[row * vocab_size + token as usize] = 10.0;
        }

        Ok(Tensor::from_f32(
            &[batch.num_sequences(), vocab_size as i32],
            &logits,
        )?)
    }

    fn is_stop_token(&self, token_id: i32) -> bool {
        token_id == STOP_TOKEN
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn output_dim(&self) -> i32 {
        VOCAB.len() as i32
    }

    fn kv_cache_spec(&self) -> llm::Result<KVCacheSpec> {
        Ok(MockModel::spec())
    }

    fn build_prompt(&self, history: &[llm::Message]) -> llm::Result<Prompt> {
        let mut prompt = Prompt::new();
        for message in history {
            prompt.append_text(&message.content);
        }
        Ok(prompt)
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

fn cache(num_blocks: i32) -> KVCacheManager {
    KVCacheManager::new(MockModel::spec(), 4, num_blocks, Device::Cpu).unwrap()
}

fn greedy(max_tokens: i32) -> GenerationConfig {
    GenerationConfig {
        top_k: 0,
        top_p: 1.0,
        temperature: 0.0,
        max_tokens,
    }
}

/// Steps until every request has had its final output, or the step budget runs out.
fn run<M: ModelForGeneration>(scheduler: &mut Scheduler<M>, steps: usize) -> Vec<RequestOutput> {
    let mut outputs = Vec::new();
    for _ in 0..steps {
        if !scheduler.has_unfinished_requests() {
            break;
        }
        outputs.extend(scheduler.step());
    }
    outputs
}

#[test]
fn generates_until_the_token_budget_runs_out() {
    let model = MockModel::new(vec![2, 3, 4]);
    let mut scheduler = Scheduler::new(model, cache(8), 16).unwrap();

    scheduler
        .add_request(Request::new("r1", vec![2, 3], greedy(3)).unwrap())
        .unwrap();
    let outputs = run(&mut scheduler, 10);

    let tokens: Vec<i64> = outputs.iter().flat_map(|o| o.token_ids.clone()).collect();
    assert_eq!(tokens, vec![2, 3, 4]);
    assert_eq!(
        outputs.iter().map(|o| o.text.as_str()).collect::<String>(),
        "abc"
    );

    let last = outputs.last().unwrap();
    assert!(last.finished);
    assert_eq!(last.finish_reason, Some(FinishReason::Length));
    assert_eq!(outputs.iter().filter(|o| o.finished).count(), 1);
    assert!(!scheduler.has_unfinished_requests());
}

#[test]
fn stops_on_a_stop_token_without_emitting_it() {
    let model = MockModel::new(vec![2, STOP_TOKEN as i64]);
    let mut scheduler = Scheduler::new(model, cache(8), 16).unwrap();

    scheduler
        .add_request(Request::new("r1", vec![2], greedy(10)).unwrap())
        .unwrap();
    let outputs = run(&mut scheduler, 10);

    let tokens: Vec<i64> = outputs.iter().flat_map(|o| o.token_ids.clone()).collect();
    assert_eq!(tokens, vec![2], "the stop token is not part of the output");
    assert_eq!(
        outputs.last().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
}

#[test]
fn chunks_a_prompt_that_is_longer_than_one_pass() {
    // A budget of four tokens against a prompt of ten: three passes to read it, and no token is
    // produced until the last of them finishes the prompt.
    let mut scheduler = Scheduler::new(MockModel::new(vec![2]), cache(8), 4).unwrap();
    scheduler
        .add_request(Request::new("r1", vec![2; 10], greedy(1)).unwrap())
        .unwrap();

    let outputs = run(&mut scheduler, 10);
    assert_eq!(
        outputs.len(),
        1,
        "one output, from the pass that finished the prompt"
    );
    assert_eq!(outputs[0].token_ids, vec![2]);

    let passes = scheduler.model().passes.lock().unwrap().clone();
    assert_eq!(
        passes,
        vec![4, 4, 2],
        "the prompt is read four tokens at a time"
    );
}

#[test]
fn batches_several_requests_into_one_pass() {
    let mut scheduler = Scheduler::new(MockModel::new(vec![2, 3]), cache(8), 16).unwrap();
    for id in ["r1", "r2", "r3"] {
        scheduler
            .add_request(Request::new(id, vec![2, 3], greedy(2)).unwrap())
            .unwrap();
    }

    let outputs = run(&mut scheduler, 10);

    // Three prompts of two tokens go in together, then the three generated tokens do.
    let passes = scheduler.model().passes.lock().unwrap().clone();
    assert_eq!(passes, vec![6, 3], "all three share every pass");

    for id in ["r1", "r2", "r3"] {
        let mine: Vec<&RequestOutput> = outputs.iter().filter(|o| o.request_id == id).collect();
        let tokens: Vec<i64> = mine.iter().flat_map(|o| o.token_ids.clone()).collect();
        assert_eq!(tokens, vec![2, 3], "{id} generated the wrong tokens");
        assert_eq!(mine.iter().filter(|o| o.finished).count(), 1);
    }
}

#[test]
fn preempts_a_running_request_when_the_cache_runs_out() {
    // Two blocks of four tokens each: enough for one request at a time, not two.
    let mut scheduler = Scheduler::new(MockModel::new(vec![2]), cache(2), 16).unwrap();
    scheduler
        .add_request(Request::new("r1", vec![2; 5], greedy(4)).unwrap())
        .unwrap();
    scheduler
        .add_request(Request::new("r2", vec![2; 5], greedy(4)).unwrap())
        .unwrap();

    let outputs = run(&mut scheduler, 40);

    // Both finish, one after the other, rather than deadlocking over the blocks.
    for id in ["r1", "r2"] {
        let finals: Vec<&RequestOutput> = outputs
            .iter()
            .filter(|o| o.request_id == id && o.finished)
            .collect();
        assert_eq!(finals.len(), 1, "{id} did not finish exactly once");
        assert_eq!(finals[0].finish_reason, Some(FinishReason::Length));
    }
}

#[test]
fn fails_a_request_that_is_larger_than_the_cache() {
    let mut scheduler = Scheduler::new(MockModel::new(vec![2]), cache(2), 64).unwrap();
    scheduler
        .add_request(Request::new("big", vec![2; 40], greedy(1)).unwrap())
        .unwrap();

    let outputs = run(&mut scheduler, 5);
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].finish_reason, Some(FinishReason::Error));
    assert!(
        outputs[0].error_message.contains("kv cache"),
        "{:?}",
        outputs[0]
    );
}

#[test]
fn cancelling_still_produces_a_final_output() {
    let mut scheduler = Scheduler::new(MockModel::new(vec![2]), cache(8), 16).unwrap();
    scheduler
        .add_request(Request::new("r1", vec![2], greedy(100)).unwrap())
        .unwrap();
    scheduler.step();

    scheduler.abort_request("r1");
    let outputs = scheduler.step();

    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].finished);
    assert_eq!(outputs[0].finish_reason, Some(FinishReason::Cancelled));
    assert!(!scheduler.has_unfinished_requests());

    // Aborting something that was never here does nothing at all.
    scheduler.abort_request("nobody");
    assert!(scheduler.step().is_empty());
}

#[test]
fn a_request_allowed_no_tokens_finishes_straight_away() {
    let mut scheduler = Scheduler::new(MockModel::new(vec![2]), cache(8), 16).unwrap();
    scheduler
        .add_request(Request::new("r1", vec![2], greedy(0)).unwrap())
        .unwrap();

    let outputs = scheduler.step();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].finish_reason, Some(FinishReason::Length));
    assert!(outputs[0].token_ids.is_empty());
    assert_eq!(scheduler.model().forwards.load(Ordering::SeqCst), 0);
}

#[test]
fn a_failed_pass_fails_the_requests_that_were_in_it() {
    let mut scheduler = Scheduler::new(MockModel::failing(), cache(8), 16).unwrap();
    scheduler
        .add_request(Request::new("r1", vec![2], greedy(4)).unwrap())
        .unwrap();
    scheduler
        .add_request(Request::new("r2", vec![2], greedy(4)).unwrap())
        .unwrap();

    let outputs = run(&mut scheduler, 5);
    assert_eq!(outputs.len(), 2);
    for output in &outputs {
        assert_eq!(output.finish_reason, Some(FinishReason::Error));
        assert!(output.error_message.contains("broke"), "{output:?}");
    }
    assert!(!scheduler.has_unfinished_requests());
}

#[test]
fn refuses_a_request_it_is_already_running() {
    let mut scheduler = Scheduler::new(MockModel::new(vec![2]), cache(8), 16).unwrap();
    scheduler
        .add_request(Request::new("r1", vec![2], greedy(4)).unwrap())
        .unwrap();

    let error = scheduler
        .add_request(Request::new("r1", vec![2], greedy(4)).unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("already running"), "{error}");

    // Sampling parameters are checked when the request is accepted, not when it is sampled.
    let bad = GenerationConfig {
        top_p: 2.0,
        ..greedy(4)
    };
    assert!(scheduler
        .add_request(Request::new("r2", vec![2], bad).unwrap())
        .is_err());
}

/// Waits for a request's final output, since generation runs on the engine's own thread.
fn wait_for_final(collected: &Arc<Mutex<Vec<RequestOutput>>>, request_id: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if collected
            .lock()
            .unwrap()
            .iter()
            .any(|o| o.request_id == request_id && o.finished)
        {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("{request_id} never produced a final output");
}

#[test]
fn the_engine_delivers_what_it_generates_to_the_callback() {
    let collected: Arc<Mutex<Vec<RequestOutput>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&collected);

    let engine = llm::Engine::new(
        || Ok((MockModel::new(vec![2, 3, 4]), cache(8))),
        16,
        move |outputs: &[RequestOutput]| sink.lock().unwrap().extend_from_slice(outputs),
    )
    .unwrap();

    engine
        .add_request(Request::new("r1", vec![2], greedy(3)).unwrap())
        .unwrap();
    wait_for_final(&collected, "r1");
    engine.shutdown().unwrap();

    let outputs = collected.lock().unwrap();
    let tokens: Vec<i64> = outputs.iter().flat_map(|o| o.token_ids.clone()).collect();
    assert_eq!(tokens, vec![2, 3, 4]);
    assert_eq!(outputs.iter().filter(|o| o.finished).count(), 1);
    assert_eq!(
        outputs.last().unwrap().finish_reason,
        Some(FinishReason::Length)
    );
}

#[test]
fn shutting_down_cancels_what_is_still_running() {
    let collected: Arc<Mutex<Vec<RequestOutput>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&collected);

    let engine = llm::Engine::new(
        || Ok((MockModel::new(vec![2]), cache(8))),
        16,
        move |outputs: &[RequestOutput]| sink.lock().unwrap().extend_from_slice(outputs),
    )
    .unwrap();

    // A request with a budget far larger than the wait below: shutting down has to cut it off
    // rather than wait for it, and still owes it a final output.
    engine
        .add_request(Request::new("long", vec![2], greedy(100_000)).unwrap())
        .unwrap();
    engine.shutdown().unwrap();

    let outputs = collected.lock().unwrap();
    let finals: Vec<&RequestOutput> = outputs.iter().filter(|o| o.finished).collect();
    assert_eq!(finals.len(), 1);
    assert_eq!(finals[0].request_id, "long");
    assert_eq!(finals[0].finish_reason, Some(FinishReason::Cancelled));

    // Nothing is accepted once it is shutting down.
    assert!(engine
        .add_request(Request::new("late", vec![2], greedy(1)).unwrap())
        .is_err());
}

#[test]
fn the_engine_reports_a_model_that_will_not_load() {
    let started = llm::Engine::new(
        || -> llm::Result<(MockModel, KVCacheManager)> {
            Err(llm::Error::Model("no such model".to_string()))
        },
        16,
        |_: &[RequestOutput]| {},
    );

    let error = started.unwrap_err();
    assert!(error.to_string().contains("no such model"), "{error}");
}

#[test]
fn the_engine_answers_a_conversation() {
    let collected: Arc<Mutex<Vec<RequestOutput>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&collected);

    let engine = llm::Engine::new(
        || Ok((MockModel::new(vec![2, 3]), cache(8))),
        16,
        move |outputs: &[RequestOutput]| sink.lock().unwrap().extend_from_slice(outputs),
    )
    .unwrap();

    // The model lays the conversation out and encodes it, on its own thread.
    engine
        .add_request_input(
            "chat",
            llm::RequestInput::Messages(vec![llm::Message::new("user", "abc")]),
            greedy(2),
        )
        .unwrap();
    wait_for_final(&collected, "chat");
    engine.shutdown().unwrap();

    let outputs = collected.lock().unwrap();
    let text: String = outputs.iter().map(|o| o.text.clone()).collect();
    assert_eq!(text, "ab");
    assert_eq!(outputs.last().unwrap().request_id, "chat");
}

#[test]
fn the_engine_owes_a_final_output_even_for_a_request_it_cannot_take() {
    let collected: Arc<Mutex<Vec<RequestOutput>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&collected);

    let engine = llm::Engine::new(
        || Ok((MockModel::new(vec![2]), cache(8))),
        16,
        move |outputs: &[RequestOutput]| sink.lock().unwrap().extend_from_slice(outputs),
    )
    .unwrap();

    // An empty conversation cannot be laid out, and the failure comes back on the callback.
    engine
        .add_request_input("bad", llm::RequestInput::Messages(vec![]), greedy(2))
        .unwrap();
    engine.shutdown().unwrap();

    let outputs = collected.lock().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].request_id, "bad");
    assert_eq!(outputs[0].finish_reason, Some(FinishReason::Error));
}

#[test]
fn sizing_the_cache_needs_a_device_that_reports_its_memory() {
    // The CPU backend reports nothing, so there is no budget to divide up and this says so
    // rather than allocating something arbitrary.
    let error =
        KVCacheManager::for_model(&MockModel::new(vec![2]), &EngineConfig::default()).unwrap_err();
    assert!(error.to_string().contains("memory"), "{error}");
}
