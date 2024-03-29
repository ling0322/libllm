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

// Deuqant qint4 (q4) to single (s).

#pragma once

#include <omp.h>
#include "libllm/cpu/kernel/dequant_common.h"
#include "libllm/cpu/kernel/kernel_hq4.h"
#include "libllm/lut/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

typedef DequantQInt4Impl<Float16, DequantQInt4ToHalfAsimdhpKernel, Mode::SingleThread> 
    DequantQInt4ToHalfAsimdhp;
typedef DequantQInt4Impl<Float16, DequantQInt4ToHalfFallbackKernel, Mode::SingleThread>
    DequantQInt4ToHalfFallback;
typedef DequantQInt4Impl<Float16, DequantQInt4ToHalfAsimdhpKernel, Mode::OMP>
    DequantQInt4ToHalfAsimdhpOMP;
typedef DequantQInt4Impl<Float16, DequantQInt4ToHalfFallbackKernel, Mode::OMP>
    DequantQInt4ToHalfFallbackOMP;

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
