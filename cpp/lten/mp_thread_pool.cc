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

#include "lten/mp.h"
#include "lutil/log.h"
#include "lutil/range.h"
#include "lutil/thread_pool.h"

namespace lten {

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

void MP::parallelFor2(int numBlocks, std::function<void(Context)> closure) {
  CHECK(gThreadPoolMP) << "call MP::parallelFor() before MP::init()";
  int n = gThreadPoolMP->getNumThreads();

  std::atomic<int> numDone{0};
  for (int i = 0; i < n; ++i) {
    gThreadPoolMP->apply([numBlocks, closure, i, n, &numDone]() {
      for (int j = i; j < numBlocks; j += n) {
        closure(Context(j, numBlocks, lut::ThreadPool::getThreadId()));
      }
      numDone.fetch_add(1);
    });
  }

  while (numDone.load() < n) {
    std::this_thread::yield();
  }
}

}  // namespace lten
