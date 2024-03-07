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

#include "libllm/cpu/kernel/args.h"
#include "libllm/cpu/kernel/gemv_kernel.h"
#include "libllm/cpu/kernel/dequant_sq4.h"
#include "libllm/cpu/kernel/kernel_sq4.h"
#include "libllm/cpu/kernel/gemm_s.h"
#include "libllm/cpu/kernel/kernel_s.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

class GemmQ4 {
 public:
  virtual ~GemmQ4() = default;
  virtual void apply(const GemmQ4Args &args) const = 0;
};

template<class TGemmKernel, class TGemvImpl, class TDequantQ4Impl>
class GemmQ4Impl : public GemmQ4 {
 public:
  void apply(const GemmQ4Args &args) const override {
    if (args.M == 1) {
      // fill C with zero.
      std::fill(args.C, args.C + args.N, 0.0f);

      TGemvImpl().apply(Q4GemvArgs{
        !args.transB,
        args.transB ? args.N : args.K,
        args.transB ? args.K : args.N,
        args.B,
        args.A,
        args.transA ? args.lda : 1,
        args.C,
        1});
    } else {
      int numelB = args.K * args.N;
      lut::c_ptr<float> B = alignedAlloc<float>(numelB);
      TDequantQ4Impl().apply(numelB, args.B, 0, B.get());

      int ldb = args.transB ? args.K : args.N;
      TGemmKernel().apply(
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

typedef GemvKernel<Q4GemvArgs, AxpyQ4NotImplKernel, DotQ4Avx2Kernel, Mode::SingleThread>
        Q4GemvImplAvx2;
typedef GemvKernel<Q4GemvArgs, AxpyQ4NotImplKernel, DotQ4Avx2Kernel, Mode::OMP>
        Q4GemvImplAvx2OMP;
typedef GemvKernel<Q4GemvArgs, AxpyQ4NotImplKernel, DotQ4FallbackKernel, Mode::SingleThread>
        Q4GemvImplFallback;
typedef GemvKernel<Q4GemvArgs, AxpyQ4NotImplKernel, DotQ4FallbackKernel, Mode::OMP>
        Q4GemvImplFallbackOMP;
  
typedef GemmQ4Impl<SGEMMImplAvx512, Q4GemvImplAvx2, DequantQ4Avx2> Q4GemmAvx512;
typedef GemmQ4Impl<SGEMMImplAvx2, Q4GemvImplAvx2, DequantQ4Avx2> Q4GemmAvx2;
typedef GemmQ4Impl<SGEMMImplDefault, Q4GemvImplFallback, DequantQ4Fallback> Q4GemmFallback;


typedef GemmQ4Impl<SGEMMImplAvx512OMP, Q4GemvImplAvx2OMP, DequantQ4Avx2OMP> Q4GemmAvx512OMP;
typedef GemmQ4Impl<SGEMMImplAvx2OMP, Q4GemvImplAvx2OMP, DequantQ4Avx2OMP> Q4GemmAvx2OMP;
typedef GemmQ4Impl<SGEMMImplDefaultOMP, Q4GemvImplFallbackOMP, DequantQ4FallbackOMP>
    Q4GemmFallbackOMP;

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
