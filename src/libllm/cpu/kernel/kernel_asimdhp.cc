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
#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/cpu/kernel/kernel_h.h"
#include "libllm/cpu/kernel/kernel_hq4.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

LIBLLM_KERNEL_FORCE_INLINE _Float16 hsum(float16x8_t q0) {
  q0 = vpaddq_f16(q0, q0);
  q0 = vpaddq_f16(q0, q0);
  q0 = vpaddq_f16(q0, q0);

  return vgetq_lane_f16(q0, 0);
}

void AxpyHalfAsimdhpKernel::apply(int64_t n, Float16 a, const Float16 *x, float *y) {
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

void AxpyHalfAsimdhpKernel::applyColumn(const GemvArgs<Float16> &args, int column, float *y) {
  apply(args.N, args.x[column], args.A + column * args.lda, y);
}

#define LIBLLM_DotHalfAsimdhpKernel_FmaBlock \
    x00 = vld1q_f16(px); \
    y00 = vld1q_f16(py); \
    ha00 = vfmaq_f16(ha00, x00, y00); \
    px += 8; \
    py += 8;

Float16 DotHalfAsimdhpKernel::apply(int64_t n, const Float16 *x, const Float16 *y) {
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

    LIBLLM_DotHalfAsimdhpKernel_FmaBlock
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock
    LIBLLM_DotHalfAsimdhpKernel_FmaBlock

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

Float16 DotHalfAsimdhpKernel::applyRow(const GemvArgs<Float16> &args, int row) {
  return apply(args.N, args.A + row * args.lda, args.x);
}

#define LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(m, r, rl) \
    a00 = vdupq_n_f16(vget_lane_f16(ra0 ## r, rl)); \
    c ## m ## 0 = vfmaq_f16(c ## m ## 0, a00, b00); \
    c ## m ## 1 = vfmaq_f16(c ## m ## 1, a00, b01);

#define LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(m) \
    a00 = vld1q_f16(pc); \
    c##m##0 = vaddq_f16(a00, c##m##0); \
    vst1q_f16(pc, c##m##0); \
    a00 = vld1q_f16(pc + 8); \
    c##m##1 = vaddq_f16(a00, c##m##1); \
    vst1q_f16(pc + 8, c##m##1); \
    pc += rs_c;

void GemmHalf12x16AsimdhpKernel::apply(
    int64_t kc, Float16 *a, Float16 *b, Float16 *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR
  float16x8_t c00 = vdupq_n_f16(0),
              c01 = vdupq_n_f16(0),
              c10 = vdupq_n_f16(0),
              c11 = vdupq_n_f16(0),
              c20 = vdupq_n_f16(0),
              c21 = vdupq_n_f16(0),
              c30 = vdupq_n_f16(0),
              c31 = vdupq_n_f16(0),
              c40 = vdupq_n_f16(0),
              c41 = vdupq_n_f16(0),
              c50 = vdupq_n_f16(0),
              c51 = vdupq_n_f16(0),
              c60 = vdupq_n_f16(0),
              c61 = vdupq_n_f16(0),
              c70 = vdupq_n_f16(0),
              c71 = vdupq_n_f16(0),
              c80 = vdupq_n_f16(0),
              c81 = vdupq_n_f16(0),
              c90 = vdupq_n_f16(0),
              c91 = vdupq_n_f16(0),
              ca0 = vdupq_n_f16(0),
              ca1 = vdupq_n_f16(0),
              cb0 = vdupq_n_f16(0),
              cb1 = vdupq_n_f16(0);
  float16x8_t a00, b00, b01;
  float16x4_t ra00, ra01, ra02;


  const __fp16 *pa = reinterpret_cast<const __fp16 *>(a);
  const __fp16 *pb = reinterpret_cast<const __fp16 *>(b);

  for (int64_t k = 0; k < kc; ++k) {
    ra00 = vld1_f16(pa); pa += 4;
    ra01 = vld1_f16(pa); pa += 4;
    ra02 = vld1_f16(pa); pa += 4;

    b00 = vld1q_f16(pb); pb += 8;
    b01 = vld1q_f16(pb); pb += 8;

    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(0, 0, 0)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(1, 0, 1)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(2, 0, 2)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(3, 0, 3)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(4, 1, 0)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(5, 1, 1)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(6, 1, 2)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(7, 1, 3)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(8, 2, 0)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(9, 2, 1)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(a, 2, 2)
    LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(b, 2, 3)
  }

  __fp16 *pc = reinterpret_cast<__fp16 *>(c);

  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(0)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(1)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(2)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(3)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(4)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(5)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(6)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(7)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(8)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(9)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(a)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdStC(b)
}

// convert 32 * QInt4 nibbles from ru0 to 32 * Int8, then substract the zero point (rz0) and store
// to ri0 and ri1.
#define LIBLLM_KernelAsimdhpCommon_ConvertInt4ToInt8(ru0, ru1, ru2, r0xf, ri0, ri1) \
    ru1 = vandq_u8(ru0, r0xf); \
    ru2 = vshrq_n_u8(ru0, 4); \
    ri0 = vreinterpretq_s8_u8(vzip1q_u8(ru1, ru2)); \
    ri1 = vreinterpretq_s8_u8(vzip2q_u8(ru1, ru2));
  
// dequant 32 * Int8 (ri[0] and ri[1]) to 32 * half (rh[0:3]) with specified zero point (rz0) and
// scale (rs0).
// NOTE: ri[0] and ri[1] will be modified.
#define LIBLLM_KernelAsimdhpCommon_DequantInt8ToHalf(ri, rh, rz0, rs0) \
    ri##0 = vsubq_s8(ri##0, rz0); \
    ri##1 = vsubq_s8(ri##1, rz0); \
    rh##0 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(ri##0))); \
    rh##1 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(ri##0))); \
    rh##2 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(ri##1))); \
    rh##3 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(ri##1))); \
    rh##0 = vmulq_f16(rh##0, s00); \
    rh##1 = vmulq_f16(rh##1, s00); \
    rh##2 = vmulq_f16(rh##2, s00); \
    rh##3 = vmulq_f16(rh##3, s00);

#define LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32 \
    u00 = vld1q_u8(px); \
    px += 16; \
    \
    LIBLLM_KernelAsimdhpCommon_ConvertInt4ToInt8(u00, u01, u02, t0xf, i00, i01) \
    LIBLLM_KernelAsimdhpCommon_DequantInt8ToHalf(i0, xf0, z00, s00) \
    \
    vst1q_f16(py, xf00); \
    py += 8; \
    vst1q_f16(py, xf01); \
    py += 8; \
    vst1q_f16(py, xf02); \
    py += 8; \
    vst1q_f16(py, xf03); \
    py += 8;


void DequantQInt4ToHalfAsimdhpKernel::apply(int n, DataQInt4 x, int64_t offsetX, Float16 *y) {
  int64_t groupIdx = offsetX / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetX % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  const uint8_t *px = reinterpret_cast<const uint8_t *>(x.getDataByGroup(groupIdx)); 
  __fp16 *py = reinterpret_cast<__fp16 *>(y); 

  uint8x16_t u00, u01, u02;
  uint8x16_t t0xf = vdupq_n_u8(0xf);
  int8x16_t i00, i01;
  int8x16_t z00;
  float16x8_t s00, xf00, xf01, xf02, xf03;

  // uint8_t * 16 -> qint4 * 32
  for (int64_t i = 0; i < nb; ++i) {
    s00 = vdupq_n_f16(x.getScaleValByGroup(groupIdx + i));
    z00 = vdupq_n_s8(x.getZeroValByGroup(groupIdx + i));

    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
  }
}

#define LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(ry0) \
    x00 = vld1q_f16(px); \
    ha00 = vfmaq_f16(ha00, x00, ry0); \
    px += 8;

#define LIBLLM_HQInt4DotAsimdhpKernel_DotQInt4x32 \
    u00 = vld1q_u8(py); \
    py += 16; \
    \
    LIBLLM_KernelAsimdhpCommon_ConvertInt4ToInt8(u00, u01, u02, t0xf, i00, i01) \
    LIBLLM_KernelAsimdhpCommon_DequantInt8ToHalf(i0, h0, z00, s00) \
    \
    LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(h00) \
    LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(h01) \
    LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(h02) \
    LIBLLM_HQInt4DotAsimdhpKernel_FmaBlock(h03)

Float16 HQInt4DotAsimdhpKernel::apply(int64_t n, const Float16 *x, DataQInt4 y, int64_t offsetY) {
  int64_t groupIdx = offsetY / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetY % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x); 
  const uint8_t *py = reinterpret_cast<const uint8_t *>(y.getDataByGroup(groupIdx));

  uint8x16_t u00, u01, u02;
  uint8x16_t t0xf = vdupq_n_u8(0xf);
  int8x16_t i00, i01;
  int8x16_t z00;
  float16x8_t x00, s00, h00, h01, h02, h03;
  float16x8_t ha00;
  float32x4_t sa00 = vdupq_n_f32(0),
              sa01 = vdupq_n_f32(0);

  for (int64_t i = 0; i < nb; ++i) {
    s00 = vdupq_n_f16(y.getScaleValByGroup(groupIdx + i));
    z00 = vdupq_n_s8(y.getZeroValByGroup(groupIdx + i));
    ha00 = vdupq_n_f16(0);

    LIBLLM_HQInt4DotAsimdhpKernel_DotQInt4x32
    LIBLLM_HQInt4DotAsimdhpKernel_DotQInt4x32
    LIBLLM_HQInt4DotAsimdhpKernel_DotQInt4x32
    LIBLLM_HQInt4DotAsimdhpKernel_DotQInt4x32

    sa00 = vaddq_f32(sa00, vcvt_f32_f16(vget_low_f16(ha00)));
    sa01 = vaddq_f32(sa01, vcvt_f32_f16(vget_high_f16(ha00)));
  }

  sa00 = vpaddq_f32(sa00, sa01);
  sa00 = vpaddq_f32(sa00, sa00);
  sa00 = vpaddq_f32(sa00, sa00);

  return vget_lane_f16(vcvt_f16_f32(sa00), 0);
}

Float16 HQInt4DotAsimdhpKernel::applyRow(const QInt4GemvArgs<Float16> &args, int row) {
  int64_t offset = row * args.N;
  return apply(args.N, args.x, args.A, offset);
}

#define LIBLLM_CvtHalfToFloatAsimdhpKernel_CvtBlock \
    x00 = vld1q_f16(px); \
    y00 = vcvt_f32_f16(vget_low_f16(x00)); \
    y01 = vcvt_f32_f16(vget_high_f16(x00)); \
    vst1q_f32(py, y00); \
    vst1q_f32(py + 4, y01); \
    px += 8; \
    py += 8;

void CvtHalfToFloatAsimdhpKernel::apply(int64_t n, const Float16 *x, float *y) {
  float16x8_t x00;
  float32x4_t y00, y01;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  float *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    LIBLLM_CvtHalfToFloatAsimdhpKernel_CvtBlock
  }

  for (int64_t i = 0; i < nr; ++i) {
    *py = *reinterpret_cast<const Float16 *>(px);
    ++py;
    ++px;
  }
}

#define LIBLLM_CvtFloatToHalfAsimdhpKernel_CvtBlock \
    x00 = vld1q_f32(px); \
    x01 = vld1q_f32(px + 4); \
    y00 = vcombine_f16(vcvt_f16_f32(x00), vcvt_f16_f32(x01)); \
    vst1q_f16(py, y00); \
    px += 8; \
    py += 8;

void CvtFloatToHalfAsimdhpKernel::apply(int64_t n, const float *x, Float16 *y) {
  float32x4_t x00, x01;
  float16x8_t y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  __fp16 *py = reinterpret_cast<__fp16 *>(y);
  for (int64_t i = 0; i < nb; ++i) {
    LIBLLM_CvtFloatToHalfAsimdhpKernel_CvtBlock
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
}  // namespace libllm
