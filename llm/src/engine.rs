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

//! Runs a [`Scheduler`] on its own thread and delivers what it generates to a callback.
//!
//! Two threads do the work. The scheduler thread is the only one that ever touches the scheduler
//! or the model, so neither needs a lock; the stream thread is the only one that runs the
//! callback, so the callback sees outputs in order and never has to be reentrant. Nothing else is
//! shared: requests reach the scheduler through one channel, outputs leave through another, and
//! closing a channel is what tells the thread behind it to stop.
//!
//! # Why the model is built on the thread
//!
//! A [`flint::Tensor`] is not `Send`: the operators keep per-device state that is not ready to be
//! used from two threads. So the engine cannot be handed a model to move onto its thread; it is
//! handed a closure that builds one, and runs it there. [`Engine::new`] waits for that closure, so
//! an engine that exists has a model, and a model that fails to load fails the call rather than
//! the first request.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::error::{Error, Result};
use crate::kv_cache::KVCacheManager;
use crate::model::ModelForGeneration;
use crate::prompt::Message;
use crate::request::{FinishReason, GenerationConfig, Request, RequestOutput};
use crate::scheduler::Scheduler;

/// How far generation may run ahead of the callback before the scheduler thread has to wait.
const MAX_QUEUED_OUTPUT_BATCHES: usize = 64;

/// What the scheduler thread is asked to do. Requests reach the scheduler only through these,
/// which is what keeps it to one thread.
enum Command {
    Add(Box<Request>),
    /// A request that still has to be encoded, which only the scheduler thread can do: the
    /// tokenizer belongs to the model, and the model belongs to that thread.
    AddInput {
        request_id: String,
        input: RequestInput,
        config: GenerationConfig,
    },
    Abort(String),
    /// Cancel everything, deliver what is still owed, and stop.
    Shutdown,
}

/// An engine: a model, a scheduler, and the two threads that drive them.
#[derive(Debug)]
pub struct Engine {
    commands: Option<SyncSender<Command>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    /// Set once shutdown has begun, so that a late request is refused rather than queued for a
    /// scheduler that is on its way out.
    shutting_down: Arc<AtomicBool>,
}

impl Engine {
    /// Build a model on the engine's own thread and start generating.
    ///
    /// `build` runs on the scheduler thread and returns the model together with the cache it will
    /// use; `max_num_batched_tokens` is the query-token budget of one forward pass. `callback`
    /// runs on the stream thread, once per step that produced anything.
    pub fn new<M, B, C>(build: B, max_num_batched_tokens: i32, callback: C) -> Result<Engine>
    where
        M: ModelForGeneration,
        B: FnOnce() -> Result<(M, KVCacheManager)> + Send + 'static,
        C: Fn(&[RequestOutput]) + Send + 'static,
    {
        let (command_tx, command_rx) = sync_channel::<Command>(MAX_QUEUED_OUTPUT_BATCHES);
        let (output_tx, output_rx) = sync_channel::<Vec<RequestOutput>>(MAX_QUEUED_OUTPUT_BATCHES);
        // Reports whether the model loaded, so that `new` can fail the way a constructor should.
        let (ready_tx, ready_rx) = sync_channel::<Result<()>>(1);

        let scheduler_thread = std::thread::Builder::new()
            .name("llm-scheduler".to_string())
            .spawn(move || {
                let scheduler = match build()
                    .and_then(|(model, cache)| Scheduler::new(model, cache, max_num_batched_tokens))
                {
                    Ok(scheduler) => {
                        let _ = ready_tx.send(Ok(()));
                        scheduler
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };

                scheduler_main(scheduler, command_rx, output_tx);
            })
            .map_err(Error::Io)?;

        let stream_thread = std::thread::Builder::new()
            .name("llm-stream".to_string())
            .spawn(move || {
                for outputs in output_rx {
                    callback(&outputs);
                }
            })
            .map_err(Error::Io)?;

        // A model that does not load is the caller's failure, not a request's.
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let _ = scheduler_thread.join();
                let _ = stream_thread.join();
                return Err(error);
            }
            Err(RecvError) => {
                let _ = scheduler_thread.join();
                let _ = stream_thread.join();
                return Err(Error::model("the engine thread stopped before it started"));
            }
        }

        Ok(Engine {
            commands: Some(command_tx),
            threads: Mutex::new(vec![scheduler_thread, stream_thread]),
            shutting_down: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Accept a request and return without waiting for it.
    ///
    /// Every accepted request produces exactly one final output on the callback, including one
    /// that later fails: a duplicate id, for instance, is reported there rather than here.
    pub fn add_request(&self, request: Request) -> Result<()> {
        self.send(Command::Add(Box::new(request)))
    }

    /// Accept a request given as tokens or as a conversation.
    ///
    /// A conversation is laid out and encoded by the model, so that happens on the engine's
    /// thread; what comes back if it cannot be is a final error output, not a failure here.
    pub fn add_request_input(
        &self,
        request_id: impl Into<String>,
        input: RequestInput,
        config: GenerationConfig,
    ) -> Result<()> {
        self.send(Command::AddInput {
            request_id: request_id.into(),
            input,
            config,
        })
    }

    /// Ask for a request to stop. An id that is unknown or already finished does nothing, and the
    /// final cancelled output is still delivered.
    pub fn abort_request(&self, request_id: impl Into<String>) -> Result<()> {
        self.send(Command::Abort(request_id.into()))
    }

    /// Cancel everything still running, deliver the outputs that are owed, and stop both threads.
    ///
    /// Cancelling rather than waiting is what keeps this bounded: a request with a large token
    /// budget cannot hold the call open. Safe to call more than once and from several threads;
    /// the later callers wait for the first. Calling it from the callback would deadlock on
    /// joining the thread the callback runs on, so it is refused there.
    pub fn shutdown(&self) -> Result<()> {
        self.shutting_down.store(true, Ordering::SeqCst);

        if let Some(commands) = &self.commands {
            // A full queue means the scheduler is busy, not that it is gone; the shutdown has to
            // get through either way.
            let _ = commands.send(Command::Shutdown);
        }

        let mut threads = self
            .threads
            .lock()
            .map_err(|_| Error::model("the engine was left in a broken state by a panic"))?;
        for thread in threads.drain(..) {
            if thread.thread().id() == std::thread::current().id() {
                return Err(Error::model(
                    "shutdown() cannot be called from the output callback",
                ));
            }
            let _ = thread.join();
        }

        Ok(())
    }

    fn send(&self, command: Command) -> Result<()> {
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err(Error::model("the engine is shutting down"));
        }

        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| Error::model("the engine is shutting down"))?;

        match commands.try_send(command) {
            Ok(()) => Ok(()),
            // A full queue is back pressure rather than a failure, so this waits.
            Err(TrySendError::Full(command)) => commands
                .send(command)
                .map_err(|_| Error::model("the engine has stopped")),
            Err(TrySendError::Disconnected(_)) => Err(Error::model("the engine has stopped")),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Dropping the sender is what lets the scheduler thread finish once it has nothing left.
        self.commands = None;
        let _ = self.shutdown();
    }
}

/// The scheduler thread: take commands, step, publish.
fn scheduler_main<M: ModelForGeneration>(
    mut scheduler: Scheduler<M>,
    commands: Receiver<Command>,
    outputs: SyncSender<Vec<RequestOutput>>,
) {
    let mut draining = false;

    loop {
        // Everything owed has been delivered, so there is nothing left to wait for. Checked
        // before blocking below, since a shutdown that arrives while requests are still running
        // is only finished once they are.
        if draining && !scheduler.has_unfinished_requests() {
            break;
        }

        // With nothing to run, wait for something to do rather than spinning. With work in hand,
        // take whatever has arrived and get on with the pass.
        if scheduler.has_unfinished_requests() {
            while let Ok(command) = commands.try_recv() {
                draining |= apply(&mut scheduler, command, &outputs);
            }
        } else {
            match commands.recv() {
                Ok(command) => {
                    draining |= apply(&mut scheduler, command, &outputs);
                }
                // Every sender is gone, so nothing more can arrive.
                Err(RecvError) => break,
            }
        }

        if scheduler.has_unfinished_requests() {
            let step = scheduler.step();
            if !step.is_empty() && outputs.send(step).is_err() {
                // Nobody is listening any more, so there is no point generating.
                scheduler.abort_all_requests();
                break;
            }
        }
    }
}

/// Applies one command. Returns whether the engine should drain and stop.
fn apply<M: ModelForGeneration>(
    scheduler: &mut Scheduler<M>,
    command: Command,
    outputs: &SyncSender<Vec<RequestOutput>>,
) -> bool {
    match command {
        Command::Add(request) => {
            let request_id = request.id().to_string();
            if let Err(error) = scheduler.add_request(*request) {
                // The request was accepted by the engine, so it is owed a final output even
                // though the scheduler would not take it.
                let _ = outputs.send(vec![RequestOutput {
                    request_id,
                    finished: true,
                    finish_reason: Some(FinishReason::Error),
                    error_message: error.to_string(),
                    ..RequestOutput::default()
                }]);
            }
            false
        }
        Command::AddInput {
            request_id,
            input,
            config,
        } => {
            let built = match input {
                RequestInput::Tokens(token_ids) => Ok(token_ids),
                RequestInput::Messages(history) => scheduler
                    .model()
                    .build_prompt(&history)
                    .and_then(|prompt| scheduler.model().encode_prompt(&prompt)),
            }
            .and_then(|token_ids| Request::new(request_id.clone(), token_ids, config))
            .and_then(|request| scheduler.add_request(request));

            if let Err(error) = built {
                let _ = outputs.send(vec![RequestOutput {
                    request_id,
                    finished: true,
                    finish_reason: Some(FinishReason::Error),
                    error_message: error.to_string(),
                    ..RequestOutput::default()
                }]);
            }
            false
        }
        Command::Abort(request_id) => {
            scheduler.abort_request(&request_id);
            false
        }
        Command::Shutdown => {
            scheduler.abort_all_requests();
            true
        }
    }
}

/// What a request was given to work from.
pub enum RequestInput {
    /// Tokens for the model to continue.
    Tokens(Vec<i64>),
    /// A conversation for the model to lay out in its own template first.
    Messages(Vec<Message>),
}
