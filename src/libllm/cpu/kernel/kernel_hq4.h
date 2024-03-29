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

// math kernels for half and qint4

#pragma once

#include <stdint.h>
#include <memory>
#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/lut/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

struct HQInt4AxpyNotImplKernel {
  static void applyColumn(const QInt4GemvArgs<Float16> &args, int col, float *y) {
    NOT_IMPL();
  }
};

struct HQInt4DotAsimdhpKernel {
  static Float16 apply(int64_t n, const Float16 *x, DataQInt4 y, int64_t offsetY);
  static Float16 applyRow(const QInt4GemvArgs<Float16> &args, int row);
};

struct HQInt4DotFallbackKernel {
  static Float16 apply(int64_t n, const Float16 *x, DataQInt4 y, int64_t offsetY);
};


struct DequantQInt4ToHalfAsimdhpKernel {
  static void apply(int n, DataQInt4 x, int64_t offsetX, Float16 *y);
};

struct DequantQInt4ToHalfFallbackKernel {
  static void apply(int n, DataQInt4 x, int64_t offsetX, Float16 *y);
};

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
