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

// MP implementation with OpenMP.

#include "lynn/mp.h"

#include <omp.h>

#include <atomic>
#include <functional>
#include <thread>

#include "lutil/log.h"
#include "lutil/thread_pool.h"

namespace libllm {

void MP::init() {
  LOG(INFO) << "OMP max_threads = " << omp_get_max_threads();
}

void MP::destroy() {
}

int MP::getMaxThreads() {
  return omp_get_max_threads();
}

void MP::parallelFor(int numBlocks, std::function<void(Context)> closure) {
#pragma omp parallel for num_threads(getMaxThreads()) schedule(dynamic, 1)
  for (int i = 0; i < numBlocks; ++i) {
    closure(Context(i, numBlocks, omp_get_thread_num()));
  }
}

}  // namespace libllm
