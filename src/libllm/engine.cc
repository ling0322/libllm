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

#include "libllm/engine.h"

#include <exception>
#include <utility>

#include "libllm/request.h"
#include "lutil/error.h"
#include "lutil/log.h"

namespace libllm {
namespace {

std::string describeError(const std::exception_ptr &error) {
  try {
    std::rethrow_exception(error);
  } catch (const std::exception &e) {
    return e.what();
  } catch (...) {
    return "unknown exception";
  }
}

}  // namespace

Engine::Engine(
    std::shared_ptr<ModelForGeneration> model,
    int maxNumBatchedTokens,
    OutputCallback callback)
    : _scheduler(std::move(model), maxNumBatchedTokens),
      _callback(std::move(callback)),
      _outputQueue(kMaxQueuedOutputBatches) {
  if (!_callback) throw lut::AbortedError("engine requires an output callback");

  _streamThread = std::thread(&Engine::streamMain, this);
  _streamThreadId = _streamThread.get_id();
  try {
    _schedulerThread = std::thread(&Engine::schedulerMain, this);
  } catch (...) {
    _outputQueue.close();
    _streamThread.join();
    throw;
  }
}

Engine::~Engine() {
  shutdown();
}

void Engine::addRequest(std::shared_ptr<Request> request) {
  if (!request) throw lut::AbortedError("cannot add a null request");

  std::string requestId = request->getId();
  if (requestId.empty()) throw lut::AbortedError("request id must not be empty");

  bool pushed = _commandQueue.push([this, request = std::move(request), requestId]() {
    try {
      _scheduler.addRequest(request);
    } catch (...) {
      // The request was accepted, so it is still owed the final output it was promised.
      publishFailedAdd(requestId, describeError(std::current_exception()));
    }
  });

  // A closed queue is the engine telling us it is on its way out.
  if (!pushed) throw lut::AbortedError("engine is shutting down");
}

void Engine::abortRequest(const std::string &requestId) {
  // Dropped once the queue is closed, which is correct: a stopped engine has already cancelled
  // everything this could refer to.
  _commandQueue.push([this, requestId]() { _scheduler.abortRequest(requestId); });
}

void Engine::shutdown() {
  if (std::this_thread::get_id() == _streamThreadId) {
    LOG(FATAL) << "Engine::shutdown() must not be called from the output callback";
  }

  std::call_once(_shutdownOnce, [this]() {
    // Queued ahead of the close, so the scheduler applies it while draining and still emits the
    // final kCancelled output that every accepted request is owed.
    _commandQueue.push([this]() { _scheduler.abortAllRequests(); });

    // Closing lets the scheduler thread apply what is left in the queue and then leave its loop.
    // It closes the output queue on the way out, which drains the stream thread and ends it, so
    // by the time both joins return every final output has reached the callback.
    _commandQueue.close();
    _schedulerThread.join();
    _streamThread.join();
  });
}

void Engine::schedulerMain() {
  Command command;
  while (true) {
    // With nothing to generate there is no reason to spin, so sleep until a command arrives. That
    // wait is also where a closed queue is noticed, which is how this thread exits. With requests
    // in flight the opposite holds: blocking would stall generation for a command that may never
    // come, so take only what is already queued.
    if (!_scheduler.hasUnfinishedRequests()) {
      if (!_commandQueue.waitPop(command)) break;
      applyCommand(std::move(command));
    }
    while (_commandQueue.tryPop(command)) applyCommand(std::move(command));

    // The commands just applied may have cancelled everything that was left to do.
    if (!_scheduler.hasUnfinishedRequests()) continue;

    std::vector<RequestOutput> outputs = _scheduler.step();
    if (!outputs.empty()) _outputQueue.push(std::move(outputs));
  }
  _outputQueue.close();
}

/// Taken by value so a command, and anything it captured, is released as soon as it has run.
void Engine::applyCommand(Command command) {
  try {
    command();
  } catch (...) {
    // A command reports its own failure where it can. This is the net for the ones that do not:
    // letting the exception escape would take the scheduler thread with it and strand every
    // other request.
    LOG(ERROR) << "engine failed to apply a command: " << describeError(std::current_exception());
  }
}

void Engine::publishFailedAdd(const std::string &requestId, const std::string &message) {
  LOG(ERROR) << "engine failed to add request " << requestId << ": " << message;

  RequestOutput output;
  output.requestId = requestId;
  output.finished = true;
  output.finishReason = RequestFinishReason::kError;
  output.errorMessage = message;

  _outputQueue.push({std::move(output)});
}

void Engine::streamMain() {
  std::vector<RequestOutput> outputs;
  while (_outputQueue.waitPop(outputs)) {
    try {
      _callback(outputs);
    } catch (const std::exception &e) {
      LOG(ERROR) << "engine output callback failed: " << e.what();
    } catch (...) {
      LOG(ERROR) << "engine output callback failed with an unknown exception";
    }
  }
}

}  // namespace libllm
