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

// MP implementation with lut::ThreadPool

#include <atomic>
#include <functional>
#include <thread>

#include "libllm/lut/log.h"
#include "libllm/lut/range.h"
#include "libllm/lut/thread_pool.h"
#include "libllm/mp.h"

namespace libllm {

lut::ThreadPool *gThreadPoolMP = nullptr;

void MP::init() {
  int numthreads = std::thread::hardware_concurrency();

  CHECK(gThreadPoolMP == nullptr);
  gThreadPoolMP = new lut::ThreadPool(numthreads);
  gThreadPoolMP->start();
}

void MP::destroy() {
  delete gThreadPoolMP;
  gThreadPoolMP = nullptr;
}

int MP::getMaxThreads() {
  return gThreadPoolMP->getNumThreads();
}

void MP::parallelFor(lut::Range range, int numThreads, std::function<void(Partition)> closure) {
  CHECK(gThreadPoolMP) << "call MP::parallelFor() before MP::init()";
  int n = numThreads > 0 ? numThreads : gThreadPoolMP->getNumThreads();

  std::atomic<int> numDone{0};
  for (int i = 0; i < n; ++i) {
    gThreadPoolMP->apply([range, closure, i, n, &numDone]() {
      closure(Partition(splitRange(range, i, n), i, n, lut::ThreadPool::getThreadId()));
      numDone.fetch_add(1);
    });
  }

  while (numDone.load() < n) {
    std::this_thread::yield();
  }
}

void MP::parallelFor(lut::Range range, std::function<void(Partition)> closure) {
  return parallelFor(range, -1, closure);
}

}  // namespace libllm
