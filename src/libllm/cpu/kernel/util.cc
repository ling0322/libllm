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

#include "libllm/cpu/kernel/util.h"

#include <math.h>

#include <algorithm>

#include "libllm/cpu/kernel/abstract.h"
#include "lutil/half.h"
#include "lutil/platform.h"

#ifdef LUT_ARCH_AARCH64
#include <arm_neon.h>
#endif

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

float cvt_h2s(Float16 vh) {
#ifdef LUT_ARCH_AARCH64
  float16x4_t a00 = vld1_dup_f16(&vh);
  float32x4_t b00 = vcvt_f32_f16(a00);
  return vgetq_lane_f32(b00, 0);
#else
  return lut::cvtsh_ss(vh.h);
#endif
}

Float16 cvt_s2h(float vf) {
#ifdef LUT_ARCH_AARCH64
  float32x4_t a00 = vld1q_dup_f32(&vf);
  float16x4_t b00 = vcvt_f16_f32(a00);
  return vget_lane_f16(b00, 0);
#else
  return Float16{lut::cvtss_sh(vf)};
#endif
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
