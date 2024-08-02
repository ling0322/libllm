// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "lut/thread_pool.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "../../third_party/concurrentqueue/blockingconcurrentqueue.h"
#include "lut/error.h"
#include "lut/log.h"

namespace lut {

thread_local int gThreadId = -1;

/// @brief Implementation of ThreadPool using a lock-free queue.
class ThreadPool::Impl {
 public:
  Impl(int numThreads)
      : _done(false),
        _numThreads(numThreads) {
  }

  ~Impl() {
    dispose();
  }

  /// @brief Main function for each worker.
  /// @param done flag indicates that the ThreadPool is going to finalize and workerMain should
  /// exit (no more tasks coming).
  static void workerMain(
      int threadId,
      moodycamel::BlockingConcurrentQueue<std::function<void()>> &closureQueue,
      const std::atomic<bool> &done) {
    gThreadId = threadId;
    moodycamel::ConsumerToken ctok(closureQueue);
    while (!done.load()) {
      std::function<void()> closure;
      if (closureQueue.wait_dequeue_timed(ctok, closure, std::chrono::milliseconds(200))) {
        closure();
      }
    }
  }

  void apply(std::function<void(void)> &&closure) {
    _closureQueue.enqueue(closure);
  }

  void start() {
    CHECK(_pool.empty()) << "ThreadPool already started.";

    LOG(INFO) << "ThreadPool started. numThreads=" << _numThreads;
    for (int i = 0; i < _numThreads; ++i) {
      _pool.emplace_back(workerMain, i, std::ref(_closureQueue), std::ref(_done));
    }
  }

  void dispose() {
    _done.store(true);
    for (std::thread &t : _pool) {
      t.join();
    }
    _pool.clear();
    LOG(INFO) << "ThreadPool finished. numThreads=" << _numThreads;
  }

  int getNumThreads() const {
    return _numThreads;
  }

 private:
  std::atomic<bool> _done;
  std::vector<std::thread> _pool;
  moodycamel::BlockingConcurrentQueue<std::function<void()>> _closureQueue;

  int _numThreads;
};

ThreadPool::ThreadPool(int numThreads)
    : _impl(std::make_unique<Impl>(numThreads)) {
}

ThreadPool::~ThreadPool() {
}

void ThreadPool::start() {
  _impl->start();
}

int ThreadPool::getNumThreads() const {
  return _impl->getNumThreads();
}

int ThreadPool::getThreadId() {
  CHECK(gThreadId >= 0) << "calling getThreadId() outside ThreadPool worker threads.";
  return gThreadId;
}

void ThreadPool::apply(std::function<void(void)> &&closure) {
  return _impl->apply(std::move(closure));
}

}  // namespace lut
