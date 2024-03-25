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
#include "libllm/cpu/kernel/interfaces.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

struct CvtHalfToFloatAvx2Kernel {
  static void apply(int64_t n, const Float16 *x, float *y);
};

struct CvtHalfToFloatFallbackKernel {
  static void apply(int64_t n, const Float16 *x, float *y);
};


struct AxpyHalfAsimdhpKernel {
  typedef Float16 ValueType;

  static void apply(int64_t n, Float16 a, const Float16 *x, Float16 *y);
  static void applyColumn(const GemvArgs<Float16> &args, int row, Float16 *y);
};

struct AxpyHalfFallbackKernel {
  typedef Float16 ValueType;
  static void apply(int64_t n, Float16 a, const Float16 *x, Float16 *y);
};

struct DotHalfAsimdhpKernel {
  typedef Float16 ValueType;

  static Float16 apply(int64_t n, const Float16 *x, const Float16 *y);
  static Float16 applyRow(const GemvArgs<Float16> &args, int row);
};

struct DotHalfFallbackKernel {
  typedef Float16 ValueType;

  static Float16 apply(int64_t n, const Float16 *x, const Float16 *y);
};

template<int MR_, int NR_>
struct GemmHalfFallbackKernel {
  static constexpr int MR = MR_;
  static constexpr int NR = NR_;
  static void apply(int64_t kc, Float16 *a, Float16 *b, Float16 *c, int64_t rs_c);
};

typedef GemmHalfFallbackKernel<12, 16> GemmHalf12x16FallbackKernel;

struct GemmHalf12x16AsimdhpKernel {
  static constexpr int MR = 12;
  static constexpr int NR = 16;
  static void apply(int64_t kc, Float16 *a, Float16 *b, Float16 *c, int64_t rs_c);
};

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
