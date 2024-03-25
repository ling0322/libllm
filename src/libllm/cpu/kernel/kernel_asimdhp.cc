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

void AxpyHalfAsimdhpKernel::apply(int64_t n, Float16 a, const Float16 *x, Float16 *y) {
  float16x8_t a00 = vld1q_dup_f16(reinterpret_cast<Float16 *>(&a));
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
    *py = vaddh_f16(*py, vmulh_f16(a,  *px));
    ++px;
    ++py;
  }
}

void AxpyHalfAsimdhpKernel::applyColumn(const GemvArgs<Float16> &args, int column, Float16 *y) {
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

#define LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(m) \
    c ## m ## 0 = vld1q_f16(pc); \
    c ## m ## 1 = vld1q_f16(pc + 8); \
    pc += rs_c;

#define LIBLLM_GemmHalf12x16AsimdhpKernel_FmaBlock(m, r, rl) \
    a00 = vdupq_n_f16(vget_lane_f16(ra0 ## r, rl)); \
    c ## m ## 0 = vfmaq_f16(c ## m ## 0, a00, b00); \
    c ## m ## 1 = vfmaq_f16(c ## m ## 1, a00, b01);

#define LIBLLM_GemmHalf12x16AsimdhpKernel_StC(m) \
    vst1q_f16(pc, c ## m ## 0); \
    vst1q_f16(pc + 8, c ## m ## 1); \
    pc += rs_c;

void GemmHalf12x16AsimdhpKernel::apply(
    int64_t kc, Float16 *a, Float16 *b, Float16 *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR
  float16x8_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71;
  float16x8_t c80, c81, c90, c91, ca0, ca1, cb0, cb1;
  float16x8_t a00, b00, b01;
  float16x4_t ra00, ra01, ra02;

  __fp16 *pc = reinterpret_cast<__fp16 *>(c);

  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(0)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(1)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(2)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(3)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(4)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(5)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(6)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(7)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(8)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(9)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(a)
  LIBLLM_GemmHalf12x16AsimdhpKernel_LdC(b)

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

  pc = reinterpret_cast<__fp16 *>(c);

  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(0)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(1)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(2)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(3)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(4)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(5)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(6)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(7)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(8)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(9)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(a)
  LIBLLM_GemmHalf12x16AsimdhpKernel_StC(b)
}

#define LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32 \
    xu00 = vld1q_u8(px); \
    px += 16; \
    \
    xu01 = vandq_u8(xu00, t0xf); \
    xu02 = vshrq_n_u8(xu00, 4); \
    x00 = vreinterpretq_s8_u8(vzip1q_u8(xu01, xu02)); \
    x01 = vreinterpretq_s8_u8(vzip2q_u8(xu01, xu02)); \
    x00 = vsubq_s8(x00, z00); \
    x01 = vsubq_s8(x01, z00); \
    \
    xf00 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(x00))); \
    xf01 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(x00))); \
    xf02 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(x01))); \
    xf03 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(x01))); \
    \
    xf00 = vmulq_f16(xf00, s00); \
    vst1q_f16(py, xf00); \
    py += 8; \
    \
    xf01 = vmulq_f16(xf01, s00); \
    vst1q_f16(py, xf01); \
    py += 8; \
    \
    xf02 = vmulq_f16(xf02, s00); \
    vst1q_f16(py, xf02); \
    py += 8; \
    \
    xf03 = vmulq_f16(xf03, s00); \
    vst1q_f16(py, xf03); \
    py += 8;


void DequantQInt4ToHalfAsimdhpKernel::apply(int n, DataQInt4 x, int64_t offsetX, Float16 *y) {
  int64_t groupIdx = offsetX / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetX % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  const uint8_t *px = reinterpret_cast<const uint8_t *>(x.getDataByGroup(groupIdx)); 
  __fp16 *py = reinterpret_cast<__fp16 *>(y); 

  uint8x16_t xu00, xu01, xu02;
  uint8x16_t t0xf = vdupq_n_u8(0xf);
  int8x16_t x00, x01;
  int8x16_t z00;
  float16x8_t s00, xf00, xf01, xf02, xf03;

  // uint8_t * 16 -> qint4 * 32
  for (int64_t i = 0; i < nb; ++i) {
    s00 = vdupq_n_f16(x.getScaleValByGroup(i));
    z00 = vdupq_n_s8(x.getZeroValByGroup(i));

    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
    LIBLLM_DequantQInt4ToHalfAsimdhpKernel_DequantBlockFloat16x32
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
