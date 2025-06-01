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

#include <stdlib.h>
#include <string.h>

#include <type_traits>

#include "lten/cpu/kernel/abstract.h"
#include "lten/cpu/kernel/util.h"
#include "lten/mp.h"
#include "lutil/c_ptr.h"

namespace lten {
namespace op {
namespace cpu {
namespace kernel {

template<typename ElementA, typename ElementB, typename ElementC, CpuMathBackend TYPE, Mode MODE>
void gemvContigousN(const GemvArgs<ElementA, ElementB, ElementC> &args) {
  if (MODE == Mode::SingleThread) {
    for (int m = 0; m < args.M; ++m) {
      args.y[m] += dotKernel<ElementC, ElementB, ElementA, TYPE>(
          args.N,
          args.x,
          args.A,
          m * args.lda);
    }
  } else if (MODE == Mode::OMP) {
    MP::parallelFor(args.M, [args](MP::Context ctx) {
      args.y[ctx.getBlockIdx()] += dotKernel<ElementC, ElementB, ElementA, TYPE>(
          args.N,
          args.x,
          args.A,
          ctx.getBlockIdx() * args.lda);
    });
  } else {
    NOT_IMPL();
  }
}

template<typename ElementA, typename ElementB, typename ElementC, CpuMathBackend TYPE, Mode MODE>
void gemvContigousT(const GemvArgs<ElementA, ElementB, ElementC> &args) {
  if (MODE == Mode::SingleThread) {
    lut::c_ptr<float> y = alignedAlloc<float>(args.N);
    memset(y.get(), 0, args.N * sizeof(float));

    for (int m = 0; m < args.M; ++m) {
      axpyKernel<ElementB, ElementA, float, TYPE>(args.N, args.x[m], args.A, m * args.lda, y.get());
    }
    for (int i = 0; i < args.N; ++i) {
      args.y[i] += y.get()[i];
    }
  } else if (MODE == Mode::OMP) {
    // initialize numThreads y buffers.
    // TODO: sfill
    lut::c_ptr<float> ys = alignedAlloc<float>(args.N * MP::getMaxThreads());
    memset(ys.get(), 0, args.N * MP::getMaxThreads() * sizeof(float));

    // compute axpy.
    MP::parallelFor(args.M, [args, &ys](MP::Context ctx) {
      int m = ctx.getBlockIdx();
      float *py = ys.get() + ctx.getAttachedThreadIdx() * args.N;
      axpyKernel<ElementB, ElementA, float, TYPE>(args.N, args.x[m], args.A, m * args.lda, py);
    });

    // accumulate ys.
    // TODO: vAdd.
    for (int p = 0; p < MP::getMaxThreads(); ++p) {
      float *py = ys.get() + p * args.N;
      for (int i = 0; i < args.N; ++i) {
        args.y[i] += py[i];
      }
    }
  } else {
    NOT_IMPL();
  }
}

template<typename ElementA, typename ElementB, typename ElementC, CpuMathBackend TYPE, Mode MODE>
void gemvContigous(const GemvArgs<ElementA, ElementB, ElementC> &args) {
  if (args.transA) {
    gemvContigousT<ElementA, ElementB, ElementC, TYPE, MODE>(args);
  } else {
    gemvContigousN<ElementA, ElementB, ElementC, TYPE, MODE>(args);
  }
}

template<typename ElementA, typename ElementB, typename ElementC, CpuMathBackend TYPE, Mode MODE>
void gemv(const GemvArgs<ElementA, ElementB, ElementC> &args) {
  if (args.incX == 1 && args.incY == 1) {
    gemvContigous<ElementA, ElementB, ElementC, TYPE, MODE>(args);
  } else {
    static_assert(std::is_same<ElementB, ElementC>::value, "upsupported element type of X and Y");
    lut::c_ptr<ElementB> packedXY = alignedAlloc<ElementB>(args.M + args.N);

    // On transposed A: dimemsion of (x, y) is (M, N)
    // On non-transposed A: dimemsion of (x, y) is (N, M)
    int dimX = args.transA ? args.M : args.N;
    int dimY = args.transA ? args.N : args.M;

    const ElementB *px = args.x;
    ElementC *py = args.y;
    if (args.incX != 1) {
      px = packedXY.get();
      copyVec<ElementB>(dimX, args.x, args.incX, packedXY.get(), 1);
    }
    if (args.incY != 1) {
      py = packedXY.get() + dimX;
      copyVec<ElementC>(dimY, args.y, args.incY, py, 1);
    }

    // apply GEMV kernel.
    GemvArgs<ElementA, ElementB, ElementC> cArgs = args;
    cArgs.x = px;
    cArgs.y = py;
    gemvContigous<ElementA, ElementB, ElementC, TYPE, MODE>(cArgs);

    if (args.incY != 1) {
      copyVec<ElementC>(dimY, py, 1, args.y, args.incY);
    }
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace lten
