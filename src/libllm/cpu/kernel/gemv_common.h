// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#pragma once

#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "libllm/lut/c_ptr.h"
#include "libllm/cpu/kernel/common.h"
#include "libllm/cpu/kernel/util.h"

namespace lymath {

template<typename TArgs, class TSAxpyKernel, class TSDotKernel, Mode MODE>
class GEMVCommon {
 public:
  void apply(const TArgs &args) const;

 private:
  void applyContigousXY(const TArgs &args) const;
  void applyContigousXYTransA(const TArgs &args) const;
  void applyContigousXYNonTransA(const TArgs &args) const;
};

template<typename TArgs, class TAxpyKernel, class TDotKernel, Mode MODE>
void GEMVCommon<TArgs, TAxpyKernel, TDotKernel, MODE>::apply(const TArgs &args) const {
  if (args.incX == 1 && args.incY == 1) {
    applyContigousXY(args);
  } else {
    lut::c_ptr<typename TArgs::VecType> packedXY = salloc(args.M + args.N);

    // On transposed A: dimemsion of (x, y) is (M, N)
    // On non-transposed A: dimemsion of (x, y) is (N, M)
    int dimX = args.transA ? args.M : args.N;
    int dimY = args.transA ? args.N : args.M;

    const typename TArgs::VecType *px = args.x;
    typename TArgs::VecType *py = args.y;
    if (args.incX != 1) {
      px = packedXY.get();
      scopy(dimX, args.x, args.incX, packedXY.get(), 1);
    }
    if (args.incY != 1) {
      py = packedXY.get() + dimX;
      scopy(dimY, args.y, args.incY, py, 1);
    }

    // apply GEMV kernel.
    TArgs cArgs = args;
    cArgs.x = px;
    cArgs.y = py;
    applyContigousXY(cArgs);

    if (args.incY != 1) {
      scopy(dimY, py, 1, args.y, args.incY);
    }
  }
}

template<typename TArgs, class TAxpyKernel, class TDotKernel, Mode MODE>
void GEMVCommon<TArgs, TAxpyKernel, TDotKernel, MODE>::applyContigousXY(const TArgs &args) const {
  if (args.transA) {
    applyContigousXYTransA(args);
  } else {
    applyContigousXYNonTransA(args);
  }
}

template<typename TArgs, class TAxpyKernel, class TDotKernel, Mode MODE>
void GEMVCommon<TArgs, TAxpyKernel, TDotKernel, MODE>::applyContigousXYNonTransA(
    const TArgs &args) const {
  if (MODE == Mode::SingleThread) {
    for (int m = 0; m < args.M; ++m) {
      args.y[m] += TDotKernel::applyRow(args, m);
    }
  } else if (MODE == Mode::OMP) {
    #pragma omp parallel for
    for (int m = 0; m < args.M; ++m) {
      args.y[m] += TDotKernel::applyRow(args, m);
    }
  } else {
    NOT_IMPL();
  }
}

template<typename TArgs, class TAxpyKernel, class TDotKernel, Mode MODE>
void GEMVCommon<TArgs, TAxpyKernel, TDotKernel, MODE>::applyContigousXYTransA(
    const TArgs &args) const {
  int mp = (args.M + GEMVMinRowsPerThread - 1) / GEMVMinRowsPerThread;
  int numThreads = std::min(mp, omp_get_max_threads());

  if (MODE == Mode::SingleThread || numThreads <= 1) {
    for (int m = 0; m < args.M; ++m) {
      TAxpyKernel::applyColumn(args, m, args.y);
    }
  } else if (MODE == Mode::OMP) {
    // initialize numThreads y buffers.
    // TODO: sfill
    lut::c_ptr<typename TArgs::VecType> ys = salloc(args.N * numThreads);
    memset(ys.get(), 0, args.N * numThreads * sizeof(typename TArgs::VecType));

    // compute axpy.
    #pragma omp parallel for num_threads(numThreads)
    for (int m = 0; m < args.M; ++m) {
      typename TArgs::VecType *py = ys.get() + omp_get_thread_num() * args.N;
      TAxpyKernel::applyColumn(args, m, py);
    }

    // accumulate ys.
    // TODO: vAdd.
    for (int p = 0; p < numThreads; ++p) {
      typename TArgs::VecType *py = ys.get() + p * args.N;
      for (int i = 0; i < args.N; ++i) {
        args.y[i] += py[i];
      }
    }
  } else {
    NOT_IMPL();
  }
}

}  // namespace libllmmath
