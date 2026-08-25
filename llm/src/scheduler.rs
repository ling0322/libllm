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

//! Continuous batching: many requests, one forward pass at a time.
//!
//! Each step packs as many requests as fit a token budget into a single pass, samples one token
//! for each of them, and hands back what they produced. Requests do not wait for each other to
//! finish: one that is still working through a long prompt shares the pass with others that are
//! already generating a token at a time.
//!
//! Two things bound a step. The token budget caps how much of a long prompt goes in at once, so
//! the rest of it is chunked into later steps; the cache blocks bound how many requests can be
//! resident at all, and when they run out a running request is preempted — its blocks are taken
//! back and its computed prefix forgotten, to be recomputed when it is admitted again. Running
//! requests are preferred to waiting ones, and once a step has preempted anything it admits no
//! new waiting request, so that the capacity it just freed goes to the batch that is already
//! running.
//!
//! This is the synchronous core: it owns the requests and the model but no threads. [`Engine`]
//! puts a thread around it.
//!
//! [`Engine`]: crate::Engine

use std::collections::{HashMap, VecDeque};

use crate::error::{Error, Result};
use crate::forward_batch::ForwardBatch;
use crate::kv_cache::KVCacheManager;
use crate::model::ModelForGeneration;
use crate::request::{FinishReason, Request, RequestOutput, RequestStatus};
use crate::sampling_batch::SamplingBatch;

/// One step's worth of work: the packed batch, and what to do with what comes back.
struct ScheduledBatch {
    batch: ForwardBatch,
    /// The requests in the batch, in the packed sequence order.
    request_ids: Vec<String>,
    /// How many tokens of each request this pass forwards.
    query_lengths: Vec<i32>,
    /// The rows to sample, which are the sequences whose prompt this pass finishes.
    sampling: SamplingBatch,
}

/// The scheduler core.
pub struct Scheduler<M: ModelForGeneration> {
    model: M,
    cache: KVCacheManager,
    max_num_batched_tokens: i32,
    /// Owns the requests. A request is in exactly one of the queues below.
    requests: HashMap<String, Request>,
    /// Every request, in the order they were accepted, which is the order outputs come out in.
    request_order: VecDeque<String>,
    running_order: VecDeque<String>,
    waiting_order: VecDeque<String>,
}

impl<M: ModelForGeneration> Scheduler<M> {
    /// `max_num_batched_tokens` is the query-token budget of one forward pass.
    pub fn new(
        model: M,
        cache: KVCacheManager,
        max_num_batched_tokens: i32,
    ) -> Result<Scheduler<M>> {
        if max_num_batched_tokens <= 0 {
            return Err(Error::model("max_num_batched_tokens must be positive"));
        }

        Ok(Scheduler {
            model,
            cache,
            max_num_batched_tokens,
            requests: HashMap::new(),
            request_order: VecDeque::new(),
            running_order: VecDeque::new(),
            waiting_order: VecDeque::new(),
        })
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    /// Take a request into the waiting queue.
    pub fn add_request(&mut self, request: Request) -> Result<()> {
        if request.id().is_empty() {
            return Err(Error::model("request id is empty"));
        }
        if self.requests.contains_key(request.id()) {
            return Err(Error::model(format!(
                "request id {:?} is already running",
                request.id()
            )));
        }
        if request.is_finished() {
            return Err(Error::model("request has already finished"));
        }

        let config = *request.config();
        if !config.temperature.is_finite() || config.temperature < 0.0 {
            return Err(Error::model("temperature must be finite and not negative"));
        }
        if config.top_k < -1 || config.top_k > self.model.vocab().vocab_size() {
            return Err(Error::model("top_k is out of range"));
        }
        if !config.top_p.is_finite() || config.top_p <= 0.0 || config.top_p > 1.0 {
            return Err(Error::model("top_p is out of range"));
        }
        if config.max_tokens < 0 {
            return Err(Error::model("max_tokens must not be negative"));
        }

        let id = request.id().to_string();
        let mut request = request;
        request.set_status(RequestStatus::Waiting);
        // A request allowed no tokens still owes one final output, so it is accepted and then
        // finished rather than rejected.
        if config.max_tokens == 0 {
            request.set_pending_finish_reason(FinishReason::Length);
        }

        self.request_order.push_back(id.clone());
        self.waiting_order.push_back(id.clone());
        self.requests.insert(id, request);
        Ok(())
    }

    /// Ask for a request to stop. Unknown and already finished ids do nothing; the final
    /// cancelled output still comes out of a later [`Scheduler::step`].
    pub fn abort_request(&mut self, request_id: &str) {
        if let Some(request) = self.requests.get_mut(request_id) {
            if !request.is_final_emitted() {
                request.set_pending_finish_reason(FinishReason::Cancelled);
                request.set_error_message("");
            }
        }
    }

    /// Ask every unfinished request to stop.
    pub fn abort_all_requests(&mut self) {
        for request in self.requests.values_mut() {
            if !request.is_final_emitted() {
                request.set_pending_finish_reason(FinishReason::Cancelled);
                request.set_error_message("");
            }
        }
    }

    /// Whether any accepted request still owes a final output.
    pub fn has_unfinished_requests(&self) -> bool {
        !self.requests.is_empty()
    }

    pub fn num_unfinished_requests(&self) -> usize {
        self.requests.len()
    }

    /// Schedule and run at most one forward pass.
    ///
    /// What comes back are deltas: each entry holds only what that request produced since its
    /// last one. A step may return nothing but cancellations, without running the model at all.
    pub fn step(&mut self) -> Vec<RequestOutput> {
        let mut outputs = self.finish_cancelled_requests();
        self.remove_finished_requests();

        let batch = self.schedule();
        // Scheduling can fail a request outright, by finding it larger than the cache can ever
        // hold; those finals come out of this step too.
        outputs.append(&mut self.finish_cancelled_requests());
        self.remove_finished_requests();

        if let Some(batch) = batch {
            outputs.append(&mut self.execute(batch));
            self.remove_finished_requests();
        }

        outputs
    }

    /// Choose what runs this step and pack it.
    fn schedule(&mut self) -> Option<ScheduledBatch> {
        let mut token_budget = self.max_num_batched_tokens;

        let mut request_ids: Vec<String> = Vec::new();
        let mut query_lengths: Vec<i32> = Vec::new();
        let mut sample_sequence_indices: Vec<i64> = Vec::new();
        let mut temperatures: Vec<f32> = Vec::new();
        let mut top_ks: Vec<i32> = Vec::new();
        let mut top_ps: Vec<f32> = Vec::new();
        let mut token_ids: Vec<i64> = Vec::new();
        let mut position_ids: Vec<i64> = Vec::new();
        let mut cu_seqlens_q: Vec<i32> = vec![0];
        let mut cu_seqlens_k: Vec<i32> = vec![0];
        let mut block_ids: Vec<Vec<i32>> = Vec::new();

        // Running requests come first, then the waiting queue in the order it was joined. The
        // boundary is taken now so that requests moving between queues below do not change what
        // this step considers already running.
        let mut candidates: Vec<String> = self.running_order.iter().cloned().collect();
        let num_running_candidates = candidates.len();
        candidates.extend(self.waiting_order.iter().cloned());

        let mut preempted: Vec<String> = Vec::new();

        for (index, request_id) in candidates.iter().enumerate() {
            if token_budget == 0 {
                break;
            }

            let was_running = index < num_running_candidates;
            // A request preempted while making room for an earlier candidate is out for this
            // step, and picks up from the waiting queue on the next one.
            if preempted.contains(request_id) {
                continue;
            }
            // Once anything has been preempted, the room it freed is for the batch that is
            // already running, not for a new arrival.
            if !was_running && !preempted.is_empty() {
                break;
            }

            let request = self
                .requests
                .get(request_id)
                .expect("the queues only hold ids of requests the map owns");
            // Requests on their way out are finished elsewhere and never enter a batch.
            if request.pending_finish_reason() != FinishReason::None || request.is_final_emitted() {
                continue;
            }

            let context_length = request.context_length();
            let num_uncomputed = request.token_ids().len() as i32 - context_length;
            if num_uncomputed <= 0 {
                let request = self.requests.get_mut(request_id).expect("looked up above");
                request.set_pending_finish_reason(FinishReason::Error);
                request.set_error_message("request has no token to compute");
                continue;
            }

            // A long prompt is chunked to what is left of the budget. A request that is already
            // generating has one uncomputed token and so takes one slot.
            let query_length = num_uncomputed.min(token_budget);

            // Make room, evicting the lowest-priority running requests until this one fits. A
            // request may end up evicting itself, in which case it waits for a later step.
            let mut reserved = self.reserve_blocks(request_id, context_length + query_length);
            while !reserved
                && was_running
                && self.requests[request_id].pending_finish_reason() == FinishReason::None
            {
                let Some(victim_id) = self.select_preemption_victim(&request_ids) else {
                    break;
                };
                let preempted_itself = victim_id == *request_id;
                self.preempt_request(&victim_id);
                preempted.push(victim_id);

                if preempted_itself {
                    break;
                }
                reserved = self.reserve_blocks(request_id, context_length + query_length);
            }

            if !reserved {
                // A request too large for the cache to ever hold already carries its error. A
                // waiting one stays at the head of its queue until blocks come back, and one that
                // just gave up its own blocks waits a step rather than taking them straight back.
                if self.requests[request_id].pending_finish_reason() != FinishReason::None {
                    continue;
                }
                if !was_running || preempted.contains(request_id) {
                    break;
                }
                continue;
            }

            if !was_running {
                if let Some(at) = self.waiting_order.iter().position(|id| id == request_id) {
                    self.waiting_order.remove(at);
                }
                self.running_order.push_back(request_id.clone());
                self.requests
                    .get_mut(request_id)
                    .expect("looked up above")
                    .set_status(RequestStatus::Running);
            }

            let request = &self.requests[request_id];
            let query_tokens = &request.token_ids()
                [context_length as usize..(context_length + query_length) as usize];
            token_ids.extend_from_slice(query_tokens);
            position_ids.extend((0..query_length).map(|i| i64::from(context_length + i)));

            // Only a pass that finishes a request's prompt produces a token for it; the earlier
            // chunks of a long prompt only fill the cache.
            if query_length == num_uncomputed {
                let config = request.config();
                sample_sequence_indices.push(request_ids.len() as i64);
                temperatures.push(config.temperature);
                top_ks.push(config.top_k);
                top_ps.push(config.top_p);
            }

            // The query covers this step's tokens; the keys cover the cached prefix as well.
            cu_seqlens_q.push(cu_seqlens_q.last().expect("seeded with zero") + query_length);
            cu_seqlens_k.push(
                cu_seqlens_k.last().expect("seeded with zero") + context_length + query_length,
            );
            block_ids.push(request.block_ids().to_vec());
            request_ids.push(request_id.clone());
            query_lengths.push(query_length);
            token_budget -= query_length;
        }

        if request_ids.is_empty() {
            return None;
        }

        let batch = ForwardBatch::packed(token_ids, cu_seqlens_q, cu_seqlens_k, position_ids)
            .and_then(|mut batch| {
                batch.set_block_ids(block_ids)?;
                Ok(batch)
            });
        let sampling = SamplingBatch::new(sample_sequence_indices, temperatures, top_ks, top_ps);

        match (batch, sampling) {
            (Ok(batch), Ok(sampling)) => Some(ScheduledBatch {
                batch,
                request_ids,
                query_lengths,
                sampling,
            }),
            // Packing a batch out of what was just chosen should not fail; if it does, the
            // requests in it are the ones that cannot run.
            (batch, sampling) => {
                let message = batch
                    .err()
                    .map(|error| error.to_string())
                    .or_else(|| sampling.err().map(|error| error.to_string()))
                    .unwrap_or_default();
                self.fail_requests(&request_ids, &message);
                None
            }
        }
    }

    /// Run the pass and turn what comes back into outputs.
    fn execute(&mut self, batch: ScheduledBatch) -> Vec<RequestOutput> {
        let ScheduledBatch {
            batch,
            request_ids,
            query_lengths,
            sampling,
        } = batch;

        let device = self.model.device();
        let block_size = self.cache.block_size();

        let sampled = match self.run_model(batch, &sampling, device, block_size) {
            Ok(sampled) => sampled,
            Err(error) => {
                // Every request in a pass that failed fails with it: there is no way to tell
                // which of them the failure belonged to.
                self.fail_requests(&request_ids, &error.to_string());
                return self.finish_cancelled_requests();
            }
        };

        // The cache now holds these tokens, whether or not anything was sampled for them.
        for (request_id, &query_length) in request_ids.iter().zip(&query_lengths) {
            let request = self.requests.get_mut(request_id).expect("scheduled above");
            let context_length = request.context_length() + query_length;
            let advanced = request
                .advance_computed_tokens(query_length)
                .and_then(|()| request.set_context_length(context_length));
            if let Err(error) = advanced {
                request.set_pending_finish_reason(FinishReason::Error);
                request.set_error_message(error.to_string());
            }
        }

        let mut outputs = Vec::new();
        for (sample_index, &sequence_index) in sampling.sequence_indices().iter().enumerate() {
            let request_id = &request_ids[sequence_index as usize];
            let token_id = sampled[sample_index];
            outputs.push(self.accept_token(request_id, token_id));
        }

        outputs
    }

    /// Forwards the batch and samples, keeping the failure paths in one place.
    fn run_model(
        &mut self,
        batch: ForwardBatch,
        sampling: &SamplingBatch,
        device: crate::flint::Device,
        block_size: i32,
    ) -> Result<Vec<i64>> {
        let prepared = batch.prepare(device, block_size)?;
        let logits = self.model.forward(&prepared, &mut self.cache)?;

        if sampling.is_empty() {
            return Ok(Vec::new());
        }

        // A model whose logits are wider than its vocabulary has padding on the end that must not
        // be sampled from.
        let vocab_size = self.model.vocab().vocab_size();
        let width = logits.shape_at(1)?;
        if vocab_size <= 0 || vocab_size > width {
            return Err(Error::model(format!(
                "model produced {width} logits for a vocabulary of {vocab_size}"
            )));
        }
        let logits = if width == vocab_size {
            logits
        } else {
            logits.slice(1, 0, vocab_size)?.contiguous()?
        };

        let sampled = sampling.prepare(device)?.sample(&logits)?;
        Ok(sampled.to_device(crate::flint::Device::Cpu)?.to_vec_i64()?)
    }

    /// Takes one sampled token for one request and says what it produced.
    fn accept_token(&mut self, request_id: &str, token_id: i64) -> RequestOutput {
        let vocab_size = self.model.vocab().vocab_size();
        let is_stop = token_id >= 0
            && token_id < i64::from(vocab_size)
            && self.model.is_stop_token(token_id as i32);
        let piece = if is_stop || token_id < 0 || token_id >= i64::from(vocab_size) {
            String::new()
        } else {
            self.model
                .vocab()
                .decode_pieces(&[token_id as i32])
                .unwrap_or_default()
        };

        let request = self
            .requests
            .get_mut(request_id)
            .expect("a sampled request is still owned");
        let mut output = RequestOutput {
            request_id: request_id.to_string(),
            ..RequestOutput::default()
        };

        if token_id < 0 || token_id >= i64::from(vocab_size) {
            output.finished = true;
            output.finish_reason = Some(FinishReason::Error);
            output.error_message = format!(
                "sampler returned token {token_id}, which is outside a vocabulary of {vocab_size}"
            );
        } else if is_stop {
            output.finished = true;
            output.finish_reason = Some(FinishReason::Stop);
        } else {
            request.append_token(token_id);
            output.token_ids.push(token_id);
            output.text = piece;
            if request.num_generated_tokens() >= request.config().max_tokens {
                output.finished = true;
                output.finish_reason = Some(FinishReason::Length);
            }
        }

        if output.finished {
            request.finish();
            request.set_final_emitted(true);
            let blocks = request.take_block_ids();
            let _ = self.cache.free_blocks(&blocks);
        }

        output
    }

    /// The final output of every request that is on its way out, in the order they were accepted.
    fn finish_cancelled_requests(&mut self) -> Vec<RequestOutput> {
        let mut outputs = Vec::new();

        for request_id in self.request_order.clone() {
            let Some(request) = self.requests.get_mut(&request_id) else {
                continue;
            };
            if request.is_final_emitted() || request.pending_finish_reason() == FinishReason::None {
                continue;
            }

            let output = RequestOutput {
                request_id: request_id.clone(),
                finished: true,
                finish_reason: Some(request.pending_finish_reason()),
                error_message: request.error_message().to_string(),
                ..RequestOutput::default()
            };

            request.finish();
            request.set_final_emitted(true);
            let blocks = request.take_block_ids();
            let _ = self.cache.free_blocks(&blocks);

            outputs.push(output);
        }

        outputs
    }

    /// Make sure a request holds blocks for `num_tokens` tokens. False when the cache has none to
    /// spare; a request too large to ever fit is failed outright.
    fn reserve_blocks(&mut self, request_id: &str, num_tokens: i32) -> bool {
        let needed = self.cache.num_blocks_for_tokens(num_tokens);
        let too_large = num_tokens > self.cache.spec().max_context_length
            || needed > self.cache.max_num_blocks_per_request()
            || needed > self.cache.num_blocks();

        let held = self.requests[request_id].block_ids().len() as i32;
        if too_large {
            let request = self.requests.get_mut(request_id).expect("looked up above");
            request.set_pending_finish_reason(FinishReason::Error);
            request.set_error_message("request is larger than the kv cache can hold");
            return false;
        }
        if needed <= held {
            return true;
        }

        match self.cache.allocate_blocks(needed - held) {
            Some(blocks) => {
                self.requests
                    .get_mut(request_id)
                    .expect("looked up above")
                    .add_block_ids(&blocks);
                true
            }
            None => false,
        }
    }

    /// The running request to evict: the lowest-priority one that is not already in this batch.
    fn select_preemption_victim(&self, scheduled: &[String]) -> Option<String> {
        self.running_order
            .iter()
            .rev()
            .find(|id| {
                let request = &self.requests[*id];
                request.pending_finish_reason() == FinishReason::None
                    && !request.is_final_emitted()
                    && !scheduled.contains(id)
            })
            .cloned()
    }

    /// Take a running request's blocks back and send it to the front of the waiting queue. What
    /// it had computed is forgotten and recomputed when it is admitted again.
    fn preempt_request(&mut self, request_id: &str) {
        let request = self
            .requests
            .get_mut(request_id)
            .expect("a victim is a running request");
        let blocks = request.take_block_ids();
        let _ = self.cache.free_blocks(&blocks);

        let request = self.requests.get_mut(request_id).expect("looked up above");
        request.reset_computed_tokens();
        request.set_status(RequestStatus::Preempted);

        if let Some(at) = self.running_order.iter().position(|id| id == request_id) {
            self.running_order.remove(at);
        }
        self.waiting_order.push_front(request_id.to_string());
    }

    /// Mark every request of a failed pass as failed.
    fn fail_requests(&mut self, request_ids: &[String], message: &str) {
        for request_id in request_ids {
            if let Some(request) = self.requests.get_mut(request_id) {
                request.set_pending_finish_reason(FinishReason::Error);
                request.set_error_message(message);
            }
        }
    }

    /// Forget the requests whose final output has gone out.
    fn remove_finished_requests(&mut self) {
        let finished: Vec<String> = self
            .request_order
            .iter()
            .filter(|id| {
                self.requests
                    .get(*id)
                    .map(Request::is_final_emitted)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        for id in finished {
            self.requests.remove(&id);
            self.request_order.retain(|other| other != &id);
            self.running_order.retain(|other| other != &id);
            self.waiting_order.retain(|other| other != &id);
        }
    }
}

impl<M: ModelForGeneration> Drop for Scheduler<M> {
    fn drop(&mut self) {
        // Hand the blocks back, so that a scheduler that is dropped with requests still in flight
        // leaves the cache as it found it.
        for request in self.requests.values_mut() {
            let blocks = request.take_block_ids();
            let _ = self.cache.free_blocks(&blocks);
        }
    }
}
