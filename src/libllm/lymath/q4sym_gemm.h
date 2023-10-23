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

#include <stdint.h>
#include <memory>
#include "lymath/args.h"
#include "lymath/gemm_common.h"
#include "lymath/gemv_common.h"
#include "lymath/lymath.h"
#include "lymath/q4sym_dequant.h"
#include "lymath/skernel.h"
#include "lymath/sgemm.h"
#include "lymath/util.h"
#include "lyutil/time.h"

namespace lymath {

class QGemmNQNInt4A {
 public:
  virtual ~QGemmNQNInt4A() = default;
  virtual void apply(const QGemmNQNInt4AArgs &args) const = 0;
};

template<class TGEMMKernel, class TGEMVImpl, class DequantQ4SymImpl>
class QGemmNQNInt4AImpl : public QGemmNQNInt4A {
 public:
  void apply(const QGemmNQNInt4AArgs &args) const override {
    if (args.M == 1) {
      // fill C with zero.
      std::fill(args.C, args.C + args.N, 0.0f);

      TGEMVImpl().apply(QGEMVInt4AArgs{
        !args.transB,
        args.transB ? args.N : args.K,
        args.transB ? args.K : args.N,
        args.B,
        args.scaleB,
        args.A,
        args.transA ? args.lda : 1,
        args.C,
        1});
    } else {
      int numelB = args.K * args.N;
      ly::c_ptr<float> B = salloc(numelB);
      DequantQ4SymImpl().apply(numelB, args.B, args.scaleB, B.get());

      int64_t ldb = args.transB ? args.K : args.N;
      TGEMMKernel().apply(
          args.transA,
          args.transB,
          args.M,
          args.N,
          args.K,
          args.A,
          args.lda,
          B.get(),
          ldb, 
          args.C,
          args.ldc);
    }
  }
};

typedef GEMVCommon<QGEMVInt4AArgs,
                   AxpyQ4SymAvx2Kernel,
                   DotQ4SymAvx2Kernel,
                   Mode::SingleThread>
        QGEMVInt4AImplAvx2;

typedef GEMVCommon<QGEMVInt4AArgs,
                   AxpyQ4SymAvx2Kernel,
                   DotQ4SymAvx2Kernel,
                   Mode::OMP>
        QGEMVInt4AImplAvx2OMP;

typedef QGemmNQNInt4AImpl<SGEMMImplAvx512, QGEMVInt4AImplAvx2, DequantQ4SymAvx2> 
    QGemmNQNInt4AImplAvx512;
typedef QGemmNQNInt4AImpl<SGEMMImplAvx2, QGEMVInt4AImplAvx2, DequantQ4SymAvx2>
    QGemmNQNInt4AImplAvx2;
typedef QGemmNQNInt4AImpl<SGEMMImplDefault, QGEMVInt4AImplAvx2, DequantQ4SymAvx2>
    QGemmNQNInt4AImplFallback;


typedef QGemmNQNInt4AImpl<SGEMMImplAvx512OMP, QGEMVInt4AImplAvx2OMP, DequantQ4SymAvx2OMP> 
    QGemmNQNInt4AImplAvx512OMP;
typedef QGemmNQNInt4AImpl<SGEMMImplAvx2OMP, QGEMVInt4AImplAvx2OMP, DequantQ4SymAvx2OMP>
    QGemmNQNInt4AImplAvx2OMP;
typedef QGemmNQNInt4AImpl<SGEMMImplDefaultOMP, QGEMVInt4AImplAvx2OMP, DequantQ4SymAvx2OMP>
    QGemmNQNInt4AImplFallbackOMP;

}  // namespace lymath
