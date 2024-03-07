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


#include <assert.h>
#include <arm_neon.h>
#include <arm_fp16.h>
#include <stdint.h>
#include "libllm/cpu/kernel/args.h"
#include "libllm/cpu/kernel/common.h"
#include "libllm/cpu/kernel/kernel_half.h"


namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

void AxpyHalfAsimdhpKernel::apply(int64_t n, Fp16 a, PCFp16 x, PFp16 y) {
  float16x8_t a00 = vld1q_dup_f16(reinterpret_cast<__fp16 *>(&a));
  float16x8_t x00, y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  __fp16 * py = reinterpret_cast<__fp16 *>(y);
  for (int i = 0; i < nb; ++i) {
    x00 = vld1q_f16(px);
    y00 = vld1q_f16(py);

    y00 = vfmaq_f16(y00, x00, a00);
    vst1q_f16(py, y00);

    px += 8;
    py += 8;
  }

  for (int i = 0; i < nr; ++i) {
    *py = vaddh_f16(*py, vmulh_f16(a.v,  *px));
    ++px;
    ++py;
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
