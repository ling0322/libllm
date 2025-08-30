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

#include <arm_fp16.h>
#include <arm_neon.h>
#include <assert.h>
#include <stdint.h>

#include "lynn/cpu/kernel/abstract.h"

namespace ly {
namespace op {
namespace cpu {
namespace kernel {

void hsaxpyAsimdhpKernel(int64_t n, Float16 a, const Float16 *x, float *y) {
  float32x4_t a00 = vcvt_f32_f16(vld1_dup_f16(reinterpret_cast<__fp16 *>(&a)));
  float32x4_t x00, y00;

  int64_t nb = n / 4;
  int64_t nr = n % 4;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = vcvt_f32_f16(vld1_f16(px));
    y00 = vld1q_f32(py);

    y00 = vfmaq_f32(y00, x00, a00);
    vst1q_f32(py, y00);

    px += 4;
    py += 4;
  }

  for (int i = 0; i < nr; ++i) {
    *py += a * *px;
    ++px;
    ++py;
  }
}

#define LIBLLM_DotHalfAsimdhpKernel_FmaBlock \
  x00 = vld1q_f16(px);                       \
  y00 = vld1q_f16(py);                       \
  ha00 = vfmaq_f16(ha00, x00, y00);          \
  px += 8;                                   \
  py += 8;

Float16 hdotAsimdhpKernel(int64_t n, const Float16 *x, const Float16 *y) {
  float16x8_t x00, y00, ha00;
  float32x4_t sa00, sa01;

  sa00 = vdupq_n_f32(0);
  sa01 = vdupq_n_f32(0);

  int64_t nb = n / 64;
  int64_t nr = n % 64;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  const __fp16 *py = reinterpret_cast<const __fp16 *>(y);
  for (int i = 0; i < nb; ++i) {
    ha00 = vdupq_n_f16(0);

    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock;

    sa00 = vaddq_f32(sa00, vcvt_f32_f16(vget_low_f16(ha00)));
    sa01 = vaddq_f32(sa01, vcvt_f32_f16(vget_high_f16(ha00)));
  }

  __fp16 hsum1 = 0.0;
  for (int i = 0; i < nr; ++i) {
    hsum1 = vfmah_f16(hsum1, *px, *py);
    ++px;
    ++py;
  }
  sa00 = vaddq_f32(sa00, vcvt_f32_f16(vset_lane_f16(hsum1, vdup_n_f16(0), 0)));

  // unroll a00
  sa00 = vpaddq_f32(sa00, sa01);
  sa00 = vpaddq_f32(sa00, sa00);
  sa00 = vpaddq_f32(sa00, sa00);
  float sum0 = vgetq_lane_f32(sa00, 0);

  return vget_lane_f16(vcvt_f16_f32(vsetq_lane_f32(sum0, vdupq_n_f32(0), 0)), 0);
}

#define LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(m, r, rl) \
  c##m##0 = vfmaq_lane_f16(c##m##0, b00, ra0##r, rl);        \
  c##m##1 = vfmaq_lane_f16(c##m##1, b01, ra0##r, rl);

#define LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(m) \
  a00 = vld1q_f16(pc);                             \
  c##m##0 = vaddq_f16(a00, c##m##0);               \
  vst1q_f16(pc, c##m##0);                          \
  a00 = vld1q_f16(pc + 8);                         \
  c##m##1 = vaddq_f16(a00, c##m##1);               \
  vst1q_f16(pc + 8, c##m##1);                      \
  pc += rs_c;

void hgemm12x16AsimdhpKernel(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR
  float16x8_t c00 = vdupq_n_f16(0), c01 = vdupq_n_f16(0), c10 = vdupq_n_f16(0),
              c11 = vdupq_n_f16(0), c20 = vdupq_n_f16(0), c21 = vdupq_n_f16(0),
              c30 = vdupq_n_f16(0), c31 = vdupq_n_f16(0), c40 = vdupq_n_f16(0),
              c41 = vdupq_n_f16(0), c50 = vdupq_n_f16(0), c51 = vdupq_n_f16(0),
              c60 = vdupq_n_f16(0), c61 = vdupq_n_f16(0), c70 = vdupq_n_f16(0),
              c71 = vdupq_n_f16(0), c80 = vdupq_n_f16(0), c81 = vdupq_n_f16(0),
              c90 = vdupq_n_f16(0), c91 = vdupq_n_f16(0), ca0 = vdupq_n_f16(0),
              ca1 = vdupq_n_f16(0), cb0 = vdupq_n_f16(0), cb1 = vdupq_n_f16(0);
  float16x8_t a00, b00, b01;
  float16x4_t ra00, ra01, ra02;

  const __fp16 *pa = reinterpret_cast<const __fp16 *>(a);
  const __fp16 *pb = reinterpret_cast<const __fp16 *>(b);

  for (int64_t k = 0; k < kc; ++k) {
    ra00 = vld1_f16(pa);
    pa += 4;
    ra01 = vld1_f16(pa);
    pa += 4;
    ra02 = vld1_f16(pa);
    pa += 4;

    b00 = vld1q_f16(pb);
    pb += 8;
    b01 = vld1q_f16(pb);
    pb += 8;

    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(0, 0, 0);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(1, 0, 1);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(2, 0, 2);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(3, 0, 3);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(4, 1, 0);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(5, 1, 1);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(6, 1, 2);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(7, 1, 3);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(8, 2, 0);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(9, 2, 1);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(a, 2, 2);
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(b, 2, 3);
  }

  __fp16 *pc = reinterpret_cast<__fp16 *>(c);

  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(0);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(1);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(2);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(3);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(4);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(5);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(6);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(7);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(8);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(9);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(a);
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(b);
}

// convert 32 * QInt4 nibbles from ru0 to 32 * Int8, then substract the zero point (rz0) and store
// to ri0 and ri1.
#define LIBLLM_KernelAsimdhpCommon_ConvertInt4ToInt8(                                   \
    inU4x32,                                                                            \
    tempU8x16Arr2,                                                                      \
    inU8x16Val0x0f,                                                                     \
    outI8x16Arr2)                                                                       \
  tempU8x16Arr2##0 = vandq_u8(inU4x32, inU8x16Val0x0f);                                 \
  tempU8x16Arr2##1 = vshrq_n_u8(inU4x32, 4);                                            \
  outI8x16Arr2##0 = vreinterpretq_s8_u8(vzip1q_u8(tempU8x16Arr2##0, tempU8x16Arr2##1)); \
  outI8x16Arr2##1 = vreinterpretq_s8_u8(vzip2q_u8(tempU8x16Arr2##0, tempU8x16Arr2##1));

// dequant 32 * Int8 (ri[0] and ri[1]) to 32 * half (rh[0:3]) with specified zero point (rz0) and
// scale (rs0).
// NOTE: ri[0] and ri[1] will be modified.
#define LIBLLM_KernelAsimdhpCommon_DequantInt8ToHalf(                    \
    inI8x16Arr2,                                                         \
    outHx8Arr4,                                                          \
    inHx8Zero,                                                           \
    inHx8Scale)                                                          \
  outHx8Arr4##0 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(inI8x16Arr2##0)));  \
  outHx8Arr4##1 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(inI8x16Arr2##0))); \
  outHx8Arr4##2 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(inI8x16Arr2##1)));  \
  outHx8Arr4##3 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(inI8x16Arr2##1))); \
  outHx8Arr4##0 = vmulq_f16(outHx8Arr4##0, inHx8Scale);                  \
  outHx8Arr4##1 = vmulq_f16(outHx8Arr4##1, inHx8Scale);                  \
  outHx8Arr4##2 = vmulq_f16(outHx8Arr4##2, inHx8Scale);                  \
  outHx8Arr4##3 = vmulq_f16(outHx8Arr4##3, inHx8Scale);                  \
  outHx8Arr4##0 = vsubq_f16(outHx8Arr4##0, inHx8Zero);                   \
  outHx8Arr4##1 = vsubq_f16(outHx8Arr4##1, inHx8Zero);                   \
  outHx8Arr4##2 = vsubq_f16(outHx8Arr4##2, inHx8Zero);                   \
  outHx8Arr4##3 = vsubq_f16(outHx8Arr4##3, inHx8Zero);

#define LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32                          \
  xU4x32 = vld1q_u8(px->data);                                                                 \
                                                                                               \
  LIBLLM_KernelAsimdhpCommon_ConvertInt4ToInt8(xU4x32, tempU8x16r, constU8x16Val0xf, xI8x16r); \
  LIBLLM_KernelAsimdhpCommon_DequantInt8ToHalf(xI8x16r, yHx8r, zeroHx8, scaleHx8);             \
                                                                                               \
  vst1q_f16(py, yHx8r0);                                                                       \
  py += 8;                                                                                     \
  vst1q_f16(py, yHx8r1);                                                                       \
  py += 8;                                                                                     \
  vst1q_f16(py, yHx8r2);                                                                       \
  py += 8;                                                                                     \
  vst1q_f16(py, yHx8r3);                                                                       \
  py += 8;

void qhcvtAsimdhpKernel(int n, const QInt4x32 *x, int64_t offsetX, Float16 *y) {
  int64_t groupIdx = offsetX / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetX % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  const QInt4x32 *px = x + groupIdx;
  __fp16 *py = reinterpret_cast<__fp16 *>(y);

  uint8x16_t xU4x32, tempU8x16r0, tempU8x16r1;
  uint8x16_t constU8x16Val0xf = vdupq_n_u8(0xf);
  int8x16_t xI8x16r0, xI8x16r1;
  float16x8_t scaleHx8, zeroHx8, yHx8r0, yHx8r1, yHx8r2, yHx8r3;

  // uint8_t * 16 -> qint4 * 32
  for (int64_t i = 0; i < nb; ++i) {
    scaleHx8 = vdupq_n_f16(px->scale);
    zeroHx8 = vdupq_n_f16(px->zero);

    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32;

    ++px;
  }
}

#define LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(yHx8) \
  xHx8 = vld1q_f16(px);                              \
  accHx8 = vfmaq_f16(accHx8, xHx8, yHx8);            \
  px += 8;

#define LIBLLM_HQInt4DotAsimdhpKernel_DotQInt4x32                                              \
  yU4x32 = vld1q_u8(py->data);                                                                 \
                                                                                               \
  LIBLLM_KernelAsimdhpCommon_ConvertInt4ToInt8(yU4x32, tempU8x16r, constU8x16Val0xf, yI8x16r); \
  LIBLLM_KernelAsimdhpCommon_DequantInt8ToHalf(yI8x16r, yHx8r, zeroHx8, scaleHx8);             \
                                                                                               \
  LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(yHx8r0);                                              \
  LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(yHx8r1);                                              \
  LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(yHx8r2);                                              \
  LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(yHx8r3);

Float16 hqdotAsimdhpKernel(int64_t n, const Float16 *x, const QInt4x32 *y, int64_t offsetY) {
  int64_t groupIdx = offsetY / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetY % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  const QInt4x32 *py = y + groupIdx;

  uint8x16_t yU4x32, tempU8x16r0, tempU8x16r1;
  uint8x16_t constU8x16Val0xf = vdupq_n_u8(0xf);
  int8x16_t yI8x16r0, yI8x16r1;
  float16x8_t xHx8, scaleHx8, zeroHx8, yHx8r0, yHx8r1, yHx8r2, yHx8r3;
  float16x8_t accHx8;
  float32x4_t accSx4r0 = vdupq_n_f32(0), accSx4r1 = vdupq_n_f32(0);

  for (int64_t i = 0; i < nb; ++i) {
    scaleHx8 = vdupq_n_f16(py->scale);
    zeroHx8 = vdupq_n_f16(py->zero);
    accHx8 = vdupq_n_f16(0);

    LIBLLM_HQInt4DotAsimdhpKernel_DotQInt4x32;

    accSx4r0 = vaddq_f32(accSx4r0, vcvt_f32_f16(vget_low_f16(accHx8)));
    accSx4r1 = vaddq_f32(accSx4r1, vcvt_f32_f16(vget_high_f16(accHx8)));

    ++py;
  }

  accSx4r0 = vpaddq_f32(accSx4r0, accSx4r1);
  accSx4r0 = vpaddq_f32(accSx4r0, accSx4r0);
  accSx4r0 = vpaddq_f32(accSx4r0, accSx4r0);

  return vget_lane_f16(vcvt_f16_f32(accSx4r0), 0);
}

#define LIBLLM_CvtHalfToFloatAsimdhpKernel_CvtBlock \
  x00 = vld1q_f16(px);                              \
  y00 = vcvt_f32_f16(vget_low_f16(x00));            \
  y01 = vcvt_f32_f16(vget_high_f16(x00));           \
  vst1q_f32(py, y00);                               \
  vst1q_f32(py + 4, y01);                           \
  px += 8;                                          \
  py += 8;

void hscvtAsimdhpKernel(int64_t n, const Float16 *x, float *y) {
  float16x8_t x00;
  float32x4_t y00, y01;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  float *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    LIBLLM_CvtHalfToFloatAsimdhpKernel_CvtBlock;
  }

  for (int64_t i = 0; i < nr; ++i) {
    *py = *reinterpret_cast<const Float16 *>(px);
    ++py;
    ++px;
  }
}

#define LIBLLM_CvtFloatToHalfAsimdhpKernel_CvtBlock         \
  x00 = vld1q_f32(px);                                      \
  x01 = vld1q_f32(px + 4);                                  \
  y00 = vcombine_f16(vcvt_f16_f32(x00), vcvt_f16_f32(x01)); \
  vst1q_f16(py, y00);                                       \
  px += 8;                                                  \
  py += 8;

void shcvtAsimdhpKernel(int64_t n, const float *x, Float16 *y) {
  float32x4_t x00, x01;
  float16x8_t y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  __fp16 *py = reinterpret_cast<__fp16 *>(y);
  for (int64_t i = 0; i < nb; ++i) {
    LIBLLM_CvtFloatToHalfAsimdhpKernel_CvtBlock;
  }

  for (int64_t i = 0; i < nr; ++i) {
    *reinterpret_cast<Float16 *>(py) = *px;
    ++py;
    ++px;
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace ly
