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

// kernels for half type

#pragma once

#include <stdint.h>
#include <memory>
#include "libllm/cpu/kernel/args.h"
#include "libllm/cpu/kernel/common.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

struct CvtHalfToFloatAvx2Kernel {
  static void apply(int64_t n, PCFp16 x, PFp32 y);
};

struct CvtHalfToFloatFallbackKernel {
  static void apply(int64_t n, PCFp16 x, PFp32 y);
};

struct SGemm12x16AsimdhpKernel {
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static void apply(int64_t kc, PFp32 a, PFp32 b, PFp32 c, int64_t rs_c);
};

struct AxpyHalfAsimdhpKernel {
  typedef float ValueType;

  static void apply(int64_t n, Fp16 a, PCFp16 x, PFp16 y);
  static void applyColumn(const SGEMVArgs &args, int row, float *y);
};

struct AxpyHalfFallbackKernel {
  typedef Fp16 ValueType;

  static void apply(int64_t n, Fp16 a, PCFp16 x, PFp16 y);
};

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
