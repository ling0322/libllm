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

#include "libllm/lut/thread_pool.h"

#include <atomic>
#include <functional>
#include <vector>

#include "concurrentqueue/concurrentqueue.h"
#include "libllm/lut/log.h"

namespace lut {

/// @brief Implementation of ThreadPool using a lock-free queue.
class ThreadPool::Impl {
 public:
  Impl(int numThreads)
      : _numThreads(numThreads) {
  }

  ~Impl() {
    if (!_pool.empty()) {
      LOG(WARN) << "~ThreadPool() called without join().";
      join();
    }
  }

  /// @brief Main function for each worker.
  /// @param done flag indicates that the ThreadPool is going to finalize and workerMain should
  /// exit (no more tasks coming).
  static void workerMain(
      moodycamel::ConcurrentQueue<std::function<void()>> &closureQueue,
      const std::atomic<bool> &done) {
    while (!done.load()) {
      std::function<void()> closure;
      if (closureQueue.try_dequeue(closure)) {
        closure();
      } else {
        std::this_thread::yield();
      }
    }
  }

  void start() {
    CHECK(_pool.empty()) << "ThreadPool already started.";
    for (int i = 0; i < _numThreads; ++i) {
      _pool.emplace_back(workerMain, std::ref(_closureQueue), std::ref(_done));
    }
  }

  void join() {
    _done.store(true);
    for (std::thread &t : _pool) {
      t.join();
    }
    _pool.clear();
  }

  int getNumThreads() const {
    return _numThreads;
  }

  void apply(std::function<void()> &&closure) {
    CHECK(!_pool.empty()) << "ThreadPool not started.";
    _closureQueue.enqueue(std::move(closure));
  }

 private:
  moodycamel::ConcurrentQueue<std::function<void()>> _closureQueue;
  std::atomic<bool> _done;
  std::vector<std::thread> _pool;

  int _numThreads;
};

ThreadPool::ThreadPool(int numThreads)
    : _impl(std::make_unique<Impl>(numThreads)) {
}

void ThreadPool::start() {
  _impl->start();
}

void ThreadPool::join() {
  _impl->join();
}

int ThreadPool::getNumThreads() const {
  return _impl->getNumThreads();
}

void ThreadPool::apply(std::function<void()> &&closure) {
  return _impl->apply(std::move(closure));
}

}  // namespace lut
