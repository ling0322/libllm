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

#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/cpu/kernel/gemv_kernel.h"
#include "libllm/cpu/kernel/dequant_sq4.h"
#include "libllm/cpu/kernel/kernel_sq4.h"
#include "libllm/cpu/kernel/gemm_s.h"
#include "libllm/cpu/kernel/kernel_s.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {


template<class TDotKernel, Mode MODE>
using SQInt4GemvImpl = GemvKernel<QInt4GemvArgs<float>,
                                  SQInt4AxpyNotImplKernel,
                                  TDotKernel,
                                  MODE>;

typedef SQInt4GemvImpl<SQInt4DotAvx2Kernel, Mode::SingleThread> SQInt4GemvImplAvx2;
typedef SQInt4GemvImpl<SQInt4DotAvx2Kernel, Mode::OMP> SQInt4GemvImplAvx2OMP;
typedef SQInt4GemvImpl<SQInt4DotFallbackKernel, Mode::SingleThread> SQInt4GemvImplFallback;
typedef SQInt4GemvImpl<SQInt4DotFallbackKernel, Mode::OMP> SQInt4GemvImplFallbackOMP;
  
template<class TGemmKernel, class TGemvImpl, class TDequantQInt4Impl>
using SQInt4GemmImpl = QInt4GemmImpl<float, TGemmKernel, TGemvImpl, TDequantQInt4Impl>;

typedef SQInt4GemmImpl<SGEMMImplAvx512, SQInt4GemvImplAvx2, DequantQInt4Avx2> Q4GemmAvx512;
typedef SQInt4GemmImpl<SGEMMImplAvx2, SQInt4GemvImplAvx2, DequantQInt4Avx2> Q4GemmAvx2;
typedef SQInt4GemmImpl<SGEMMImplDefault, SQInt4GemvImplFallback, DequantQInt4Fallback> 
    Q4GemmFallback;

typedef SQInt4GemmImpl<SGEMMImplAvx512OMP, SQInt4GemvImplAvx2OMP, DequantQInt4Avx2OMP> 
    Q4GemmAvx512OMP;
typedef SQInt4GemmImpl<SGEMMImplAvx2OMP, SQInt4GemvImplAvx2OMP, DequantQInt4Avx2OMP> 
    Q4GemmAvx2OMP;
typedef SQInt4GemmImpl<SGEMMImplDefaultOMP, SQInt4GemvImplFallbackOMP, DequantQInt4FallbackOMP>
    Q4GemmFallbackOMP;

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
