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
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/kernel/gemm_common.h"
#include "libllm/cpu/kernel/gemm_kernel.h"
#include "libllm/cpu/kernel/gemv_s.h"
#include "libllm/cpu/kernel/kernel_s.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

// -- class SGEMM ----------

typedef GemmKernel<288, 512, 4096, float, SGemm6x16DefaultKernel, Mode::SingleThread>
    SGEMMKernelDefault;
typedef GemmKernel<288, 512, 4096, float, SGemm6x16Avx2Kernel, Mode::SingleThread>
    SGEMMKernelAvx2;
typedef GemmKernel<576, 512, 4096, float, SGemm12x32Avx512Kernel, Mode::SingleThread>
    SGEMMKernelAvx512;

typedef GemmKernel<288, 512, 4096, float, SGemm6x16DefaultKernel, Mode::OMP>
    SGEMMKernelDefaultOMP;
typedef GemmKernel<288, 512, 4096, float, SGemm6x16Avx2Kernel, Mode::OMP>
    SGEMMKernelAvx2OMP;
typedef GemmKernel<576, 512, 4096, float, SGemm12x32Avx512Kernel, Mode::OMP>
    SGEMMKernelAvx512OMP;


template<class TGemmKernel, class TGemvKernel>
using GemmFloatImpl = GemmImpl<TGemmKernel, TGemvKernel, float>;

typedef GemmFloatImpl<SGEMMKernelAvx512OMP, SGEMVImplAvx512OMP> SGEMMImplAvx512OMP;
typedef GemmFloatImpl<SGEMMKernelAvx2OMP, SGEMVImplAvx2OMP> SGEMMImplAvx2OMP;
typedef GemmFloatImpl<SGEMMKernelDefaultOMP, SGEMVImplDefaultOMP> SGEMMImplDefaultOMP;

typedef GemmFloatImpl<SGEMMKernelAvx512, SGEMVImplAvx512> SGEMMImplAvx512;
typedef GemmFloatImpl<SGEMMKernelAvx2, SGEMVImplAvx2> SGEMMImplAvx2;
typedef GemmFloatImpl<SGEMMKernelDefault, SGEMVImplDefault> SGEMMImplDefault;


}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
