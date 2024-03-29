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

#pragma once

#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/cpu/kernel/gemv_kernel.h"
#include "libllm/cpu/kernel/dequant_hq4.h"
#include "libllm/cpu/kernel/kernel_hq4.h"
#include "libllm/cpu/kernel/gemm_h.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {


template<class TDotKernel, Mode MODE>
using HQInt4GemvImpl = GemvKernel<QInt4GemvArgs<Float16>,
                                  HQInt4AxpyNotImplKernel,
                                  TDotKernel,
                                  MODE>;

typedef HQInt4GemvImpl<HQInt4DotAsimdhpKernel, Mode::SingleThread> HQInt4GemvImplAsimdhp;
typedef HQInt4GemvImpl<HQInt4DotAsimdhpKernel, Mode::OMP> HQInt4GemvImplAsimdhpOMP;
  
template<class TGemmKernel, class TGemvImpl, class TDequantQInt4Impl>
using HQInt4GemmImpl = QInt4GemmImpl<Float16, TGemmKernel, TGemvImpl, TDequantQInt4Impl>;

typedef HQInt4GemmImpl<HGemmAsimdhp, HQInt4GemvImplAsimdhp, DequantQInt4ToHalfAsimdhp>
    HQInt4GemmAsimdhp;
typedef HQInt4GemmImpl<HGemmAsimdhpOMP, HQInt4GemvImplAsimdhpOMP, DequantQInt4ToHalfAsimdhpOMP> 
    HQInt4GemmAsimdhpOMP;

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
