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
#include "libllm/cpu/kernel/common.h"
#include "libllm/lut/c_ptr.h"
#include "libllm/lut/span.h"
#include "libllm/lut/platform.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

// copy vector x to y.
template<typename T>
void copyVec(int n, const T *x, int incx, T *y, int incy) {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = x[i * incx];
  }
}
// allocate n single float and returns the holder. the memory is 32 byte aligned.
template<typename T>
lut::c_ptr<T> alignedAlloc(int64_t n) {
  return lut::c_ptr<T>(
      reinterpret_cast<T *>(lut::alloc32ByteAlignedMem(sizeof(T) * n)),
      lut::free32ByteAlignedMem);
}

float cvt_h2s(Float16 vh);
Float16 cvt_s2h(float vf);

float cvt_h2s(Fp16 vh);
Fp16 cvt_s2h(float vf);

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
