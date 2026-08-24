// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "libllm/scheduler.h"
#include "lutil/blocking_queue.h"

namespace libllm {

/// Runs a SchedulerV2 on its own thread and publishes the generated deltas through a callback.
///
/// Two threads do the work. The scheduler thread is the only one that ever touches SchedulerV2,
/// which is what lets that class stay free of locks; the stream thread is the only one that runs
/// the callback, so the callback sees outputs in order and never needs to be reentrant. Every
/// public method may be called from any thread, including from inside the callback, with the one
/// exception noted on shutdown().
///
/// The engine holds no lock of its own. The two queues carry all the state that crosses threads:
/// requests reach the scheduler only through the command queue, outputs reach the callback only
/// through the output queue, and closing a queue is what tells the thread behind it to finish.
class Engine {
 public:
  using OutputCallback = std::function<void(const std::vector<RequestOutput> &)>;

  /// `maxNumBatchedTokens` is the query-token budget of one model forward. Throws when `callback`
  /// is empty.
  Engine(
      std::shared_ptr<ModelForGeneration> model,
      int maxNumBatchedTokens,
      OutputCallback callback);

  /// Shuts the engine down, cancelling whatever is still running.
  ~Engine();

  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  /// Accept a request and return without waiting for it to generate. Throws when the request is
  /// null, its id is empty, or the engine is shutting down. Ids are expected to be unique; one
  /// that is already active is rejected by the scheduler and reported as a kError output rather
  /// than thrown from here. Every request that is accepted produces exactly one final output on
  /// the callback, including one that later fails inside the scheduler.
  void addRequest(std::shared_ptr<Request> request);

  /// Ask for a request to be cancelled. Unknown and already finished ids are no-ops. The final
  /// kCancelled output is still delivered to the callback.
  void abortRequest(const std::string &requestId);

  /// Cancel every request that has not finished, deliver the outputs that are still owed, and
  /// join both threads. Cancelling rather than waiting keeps this bounded: a request with a large
  /// token budget cannot hold the call, or the destructor that makes it, for an unbounded time.
  ///
  /// Safe to call more than once and from several threads at once; the extra callers wait for the
  /// first one to finish. Must not be called from the output callback, which would deadlock on
  /// joining the thread it runs on.
  void shutdown();

 private:
  /// A unit of work for the scheduler thread. Requests reach SchedulerV2 only through this queue,
  /// which is what serializes access to it. Each command is a closure rather than a tagged struct,
  /// so what it carries and how it reports failure stay next to the call it stands for.
  using Command = std::function<void()>;

  /// How far generation may run ahead of the callback before it has to wait. Bounding this is
  /// safe because no public method waits on the scheduler thread, so a callback that adds or
  /// aborts requests cannot deadlock against a scheduler thread that is blocked here.
  static constexpr int kMaxQueuedOutputBatches = 64;

  SchedulerV2 _scheduler;    ///< Scheduler thread only.
  OutputCallback _callback;  ///< Stream thread only.

  std::thread _schedulerThread;
  std::thread _streamThread;

  /// Captured at construction so that shutdown() can recognise a call from the callback without
  /// reading _streamThread, which another thread may be joining at the time.
  std::thread::id _streamThreadId;

  /// Makes the teardown run exactly once, and holds back any other caller until it is done.
  std::once_flag _shutdownOnce;

  lut::BlockingQueue<Command> _commandQueue;
  lut::BlockingQueue<std::vector<RequestOutput>> _outputQueue;

  void schedulerMain();
  void streamMain();
  void applyCommand(Command command);
  void publishFailedAdd(const std::string &requestId, const std::string &message);
};

}  // namespace libllm
