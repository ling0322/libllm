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

//! One completion the engine is working on.
//!
//! A request owns its tokens — the prompt's and the generated ones together — and remembers how
//! much of them the model has already forwarded into the cache. That split is what makes chunked
//! prefill and preemption work: the tokens stay, the computed prefix and the blocks holding it can
//! be given up and rebuilt.

use crate::error::{Error, Result};

/// How a request is sampled and when it stops.
#[derive(Clone, Copy, Debug)]
pub struct GenerationConfig {
    /// Keep only the `top_k` likeliest tokens. Zero or less keeps all of them.
    pub top_k: i32,
    /// Keep the likeliest tokens whose probability adds up to `top_p`.
    pub top_p: f32,
    /// Flatten or sharpen the distribution. Zero picks the likeliest token outright.
    pub temperature: f32,
    /// The most tokens to generate.
    pub max_tokens: i32,
}

impl Default for GenerationConfig {
    fn default() -> GenerationConfig {
        GenerationConfig {
            top_k: 50,
            top_p: 0.8,
            temperature: 1.0,
            max_tokens: 2048,
        }
    }
}

/// Where a request is in the scheduler's cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestStatus {
    /// Accepted, waiting for room to run.
    Waiting,
    /// Part of the batch being forwarded.
    Running,
    /// Was running, gave its blocks up so another request could run.
    Preempted,
    /// Done, and its final output has been handed over.
    Finished,
}

/// Why a request stopped producing tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// It has not stopped.
    None,
    /// The model produced a stop token.
    Stop,
    /// It reached its token budget.
    Length,
    /// Somebody cancelled it.
    Cancelled,
    /// It failed.
    Error,
}

/// What one request produced in one step.
///
/// Deltas, not totals: each output holds only what has appeared since the last one, so a caller
/// can append them as they arrive. Exactly one output per request has `finished` set.
#[derive(Clone, Debug, Default)]
pub struct RequestOutput {
    pub request_id: String,
    pub token_ids: Vec<i64>,
    pub text: String,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
    pub error_message: String,
}

/// A request and everything the scheduler tracks about it.
#[derive(Clone, Debug)]
pub struct Request {
    id: String,
    config: GenerationConfig,
    /// The prompt's tokens, followed by the generated ones.
    token_ids: Vec<i64>,
    prompt_length: usize,
    num_computed_tokens: i32,
    context_length: i32,
    block_ids: Vec<i32>,
    status: RequestStatus,
    pending_finish_reason: FinishReason,
    error_message: String,
    final_emitted: bool,
}

impl Request {
    /// A request to complete `prompt_token_ids`, which must not be empty.
    pub fn new(
        id: impl Into<String>,
        prompt_token_ids: Vec<i64>,
        config: GenerationConfig,
    ) -> Result<Request> {
        if prompt_token_ids.is_empty() {
            return Err(Error::model("a request needs a prompt"));
        }

        Ok(Request {
            id: id.into(),
            config,
            prompt_length: prompt_token_ids.len(),
            token_ids: prompt_token_ids,
            num_computed_tokens: 0,
            context_length: 0,
            block_ids: Vec::new(),
            status: RequestStatus::Waiting,
            pending_finish_reason: FinishReason::None,
            error_message: String::new(),
            final_emitted: false,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    pub fn status(&self) -> RequestStatus {
        self.status
    }

    pub fn set_status(&mut self, status: RequestStatus) {
        self.status = status;
    }

    /// The tokens generated after the prompt.
    pub fn num_generated_tokens(&self) -> i32 {
        (self.token_ids.len() - self.prompt_length) as i32
    }

    /// Why this request is about to stop, if it is. Set before the final output is handed over.
    pub fn pending_finish_reason(&self) -> FinishReason {
        self.pending_finish_reason
    }

    pub fn set_pending_finish_reason(&mut self, reason: FinishReason) {
        self.pending_finish_reason = reason;
    }

    pub fn error_message(&self) -> &str {
        &self.error_message
    }

    pub fn set_error_message(&mut self, message: impl Into<String>) {
        self.error_message = message.into();
    }

    /// Whether the one final output this request owes has been handed over.
    pub fn is_final_emitted(&self) -> bool {
        self.final_emitted
    }

    pub fn set_final_emitted(&mut self, emitted: bool) {
        self.final_emitted = emitted;
    }

    /// The prompt's tokens followed by the generated ones.
    pub fn token_ids(&self) -> &[i64] {
        &self.token_ids
    }

    pub fn num_computed_tokens(&self) -> i32 {
        self.num_computed_tokens
    }

    /// Record that the model forwarded `num_tokens` more of this request.
    pub fn advance_computed_tokens(&mut self, num_tokens: i32) -> Result<()> {
        let advanced = self.num_computed_tokens + num_tokens;
        if num_tokens < 0 || advanced > self.token_ids.len() as i32 {
            return Err(Error::model(format!(
                "cannot forward {num_tokens} more tokens of a request that holds {}",
                self.token_ids.len()
            )));
        }

        self.num_computed_tokens = advanced;
        Ok(())
    }

    /// Forget the computed prefix, after its blocks have been taken away.
    pub fn reset_computed_tokens(&mut self) {
        self.num_computed_tokens = 0;
        self.context_length = 0;
    }

    /// The leading tokens whose keys and values the cache holds. What a pass forwards on top of
    /// this is its query.
    pub fn context_length(&self) -> i32 {
        self.context_length
    }

    pub fn set_context_length(&mut self, context_length: i32) -> Result<()> {
        if context_length < 0 || context_length > self.token_ids.len() as i32 {
            return Err(Error::model(format!(
                "context length {context_length} is not within a request of {} tokens",
                self.token_ids.len()
            )));
        }

        self.context_length = context_length;
        Ok(())
    }

    /// The cache blocks this request holds, in the order its tokens fill them.
    pub fn block_ids(&self) -> &[i32] {
        &self.block_ids
    }

    pub fn add_block_ids(&mut self, block_ids: &[i32]) {
        self.block_ids.extend_from_slice(block_ids);
    }

    /// Give up the blocks. The caller hands them back to the cache.
    pub fn take_block_ids(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.block_ids)
    }

    pub fn append_token(&mut self, token_id: i64) {
        self.token_ids.push(token_id);
    }

    pub fn is_finished(&self) -> bool {
        self.status == RequestStatus::Finished
    }

    pub fn finish(&mut self) {
        self.status = RequestStatus::Finished;
    }
}
