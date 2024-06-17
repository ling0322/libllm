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

#include <pthread.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "libllm/lut/error.h"
#include "libllm/lut/log.h"

namespace lut {

thread_local int gThreadId = -1;

class Barrier {
 public:
  explicit Barrier(int num_threads)
      : num_threads_(num_threads),
        count_(0),
        generation_(0) {
  }

  void wait() {
    int gen = generation_.load();

    // 原子性增加计数器
    if (count_.fetch_add(1) == num_threads_ - 1) {
      // 所有线程都到达了栅栏，重置计数器并增加generation
      count_.store(0);
      generation_.fetch_add(1);
    } else {
      // 等待generation改变
      while (generation_.load() == gen) {
        asm("nop");
      }
    }
  }

 private:
  const int num_threads_;
  std::atomic<int> count_;
  std::atomic<int> generation_;
};

/// @brief Implementation of ThreadPool using a lock-free queue.
class ThreadPool::Impl {
 public:
  Impl(int numThreads)
      : _done(false),
        _range(0),
        _numThreads(numThreads),
        _barrier(numThreads + 1) {
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
      int threadId,
      Barrier *barrier,
      std::function<void(Range, int)> &closure,
      lut::Range &range,
      int &numPartitions,
      std::atomic<int> &numDone,
      const std::atomic<bool> &done) {
    gThreadId = threadId;
    while (!done.load()) {
      barrier->wait();

      if (threadId < numPartitions) {
        lut::Range subrange = range.getPartition(threadId, numPartitions);
        closure(subrange, threadId);
      }

      barrier->wait();
    }
    LOG(INFO) << "workerMain() DONE!";
  }

  void start() {
    CHECK(_pool.empty()) << "ThreadPool already started.";
    /*
    for (int i = 0; i < _numThreads; ++i) {
      _pool.emplace_back(
          workerMain,
          i,
          &_barrier,
          std::ref(_closure),
          std::ref(_range),
          std::ref(_numPartitions),
          std::ref(_numDone),
          std::ref(_done));
    }*/
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

  void parallelFor(Range range, std::function<void(Range, int)> closure, int numThreads) {
    int n = numThreads > 0 ? numThreads : _numThreads;

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      closure(range.getPartition(i, n), i);
    }

    /*
    _closure = closure;
    _range = range;
    _numPartitions = n;
    _barrier.wait();
    _barrier.wait();*/
  }

 private:
  std::function<void(Range, int)> _closure;
  lut::Range _range;
  int _numPartitions;
  std::atomic<bool> _done;
  std::atomic<int> _numDone;
  std::vector<std::thread> _pool;
  Barrier _barrier;

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

void ThreadPool::parallelFor(Range range, std::function<void(Range, int)> closure, int numThreads) {
  return _impl->parallelFor(range, closure, numThreads);
}

int ThreadPool::getThreadId() {
  CHECK(gThreadId >= 0) << "calling getThreadId() outside ThreadPool worker threads.";
  return gThreadId;
}

}  // namespace lut
