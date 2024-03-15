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
#include "libllm/cpu/kernel/args.h"
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/kernel/gemm_common.h"
#include "libllm/cpu/kernel/gemv_common.h"
#include "libllm/cpu/kernel/kernel_float.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

class SGEMV {
 public:
  virtual ~SGEMV() = default;
  virtual void apply(const SGEMVArgs &args) const = 0; 
};

template<class TSAxpyKernel, class TSDotKernel, Mode MODE>
class SGEMVImpl : public SGEMV {
 public:
  void apply(const SGEMVArgs &args) const override {
    GEMVCommon<SGEMVArgs, TSAxpyKernel, TSDotKernel, MODE>().apply(args);
  }
};

typedef SGEMVImpl<SAxpyAvx2Kernel, SDotAvx2Kernel, Mode::SingleThread> SGEMVImplAvx512;
typedef SGEMVImpl<SAxpyAvx2Kernel, SDotAvx2Kernel, Mode::SingleThread> SGEMVImplAvx2;
typedef SGEMVImpl<SAxpyFallbackKernel, SDotFallbackKernel, Mode::SingleThread> SGEMVImplDefault;
typedef SGEMVImpl<SAxpyAvx2Kernel, SDotAvx2Kernel, Mode::OMP> SGEMVImplAvx512OMP;
typedef SGEMVImpl<SAxpyAvx2Kernel, SDotAvx2Kernel, Mode::OMP> SGEMVImplAvx2OMP;
typedef SGEMVImpl<SAxpyFallbackKernel, SDotFallbackKernel, Mode::OMP> SGEMVImplDefaultOMP;

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
