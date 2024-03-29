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
#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/lut/half.h"
#include "libllm/lut/platform.h"

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

void quantFloatToQInt4(
    lut::Span<const float> x,
    lut::Span<UInt4x2> qdata,
    lut::Span<Float16> qscale,
    lut::Span<UInt4x2> qzero) {
  int64_t nb = x.size() / GroupSizeQInt4;
  CHECK(x.size() % GroupSizeQInt4 == 0);
  CHECK(x.size() / 2 == qdata.size());
  CHECK(nb == qscale.size());
  CHECK((nb + 1) / 2 == qzero.size());

  for (int i = 0; i < nb; ++i) {
    int begin = i * GroupSizeQInt4;
    int end = (i + 1) * GroupSizeQInt4;

    float minVal = *std::min_element(x.data() + begin, x.data() + end);
    float maxVal = *std::max_element(x.data() + begin, x.data() + end);

    float scale = (maxVal - minVal) / 15.0;
    float zeroFp32 = roundf(-minVal / scale);
    CHECK(zeroFp32 >= 0.0f && zeroFp32 <= 15.0f);
    uint8_t zero = static_cast<uint8_t>(zeroFp32);

    for (int j = 0; j < GroupSizeQInt4; j += 2) {
      float dlFp32 = roundf((x[begin + j] - minVal) / scale);
      float dhFp32 = roundf((x[begin + j + 1] - minVal) / scale);
      CHECK(dlFp32 >= 0.0f && dlFp32 <= 15.0f && dhFp32 >= 0.0f && dhFp32 <= 15.0f);

      uint8_t dl = static_cast<uint8_t>(dlFp32);
      uint8_t dh = static_cast<uint8_t>(dhFp32);
      qdata[(begin + j) / 2].b = (dh << 4) | dl;
    }

    if (i % 2 == 0) {
      qzero[i / 2].b = 0;
      qzero[i / 2].b |= zero;
    } else {
      qzero[i / 2].b |= (zero << 4);
    }

    qscale[i] = cvtf<Float16>(scale);
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm

