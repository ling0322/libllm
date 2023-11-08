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


#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include "lymath/args.h"
#include "lymath/common.h"
#include "lymath/q4kernel.h"
#include "lymath/q4sym_kernel.h"
#include "lymath/q8kernel.h"
#include "lymath/skernel.h"

namespace lymath {

void SGemm6x16Avx2Kernel::apply(int64_t kc, PFp32 a, PFp32 b, PFp32 c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR (6 x 2 ymmX)
  __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  __m256 a00, b00, b01;

  float *pc = c;
  c00 = _mm256_loadu_ps(pc);
  c01 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c10 = _mm256_loadu_ps(pc);
  c11 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c20 = _mm256_loadu_ps(pc);
  c21 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c30 = _mm256_loadu_ps(pc);
  c31 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c40 = _mm256_loadu_ps(pc);
  c41 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c50 = _mm256_loadu_ps(pc);
  c51 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  float *pa = a;
  float *pb = b;
  for (int k = 0; k < kc; ++k) {
    b00 = _mm256_loadu_ps(pb);
    b01 = _mm256_loadu_ps(pb + 8);
    a00 = _mm256_broadcast_ss(pa);

    c00 = _mm256_fmadd_ps(a00, b00, c00);
    c01 = _mm256_fmadd_ps(a00, b01, c01);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c10 = _mm256_fmadd_ps(a00, b00, c10);
    c11 = _mm256_fmadd_ps(a00, b01, c11);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c20 = _mm256_fmadd_ps(a00, b00, c20);
    c21 = _mm256_fmadd_ps(a00, b01, c21);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c30 = _mm256_fmadd_ps(a00, b00, c30);
    c31 = _mm256_fmadd_ps(a00, b01, c31);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c40 = _mm256_fmadd_ps(a00, b00, c40);
    c41 = _mm256_fmadd_ps(a00, b01, c41);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c50 = _mm256_fmadd_ps(a00, b00, c50);
    c51 = _mm256_fmadd_ps(a00, b01, c51);
    pa += 1;

    pb += 16;
  }

  pc = c;
  _mm256_storeu_ps(pc, c00);
  _mm256_storeu_ps(pc + 8, c01);
  pc += rs_c;

  _mm256_storeu_ps(pc, c10);
  _mm256_storeu_ps(pc + 8, c11);
  pc += rs_c;

  _mm256_storeu_ps(pc, c20);
  _mm256_storeu_ps(pc + 8, c21);
  pc += rs_c;

  _mm256_storeu_ps(pc, c30);
  _mm256_storeu_ps(pc + 8, c31);
  pc += rs_c;

  _mm256_storeu_ps(pc, c40);
  _mm256_storeu_ps(pc + 8, c41);
  pc += rs_c;

  _mm256_storeu_ps(pc, c50);
  _mm256_storeu_ps(pc + 8, c51);
  pc += rs_c;
}

void SAxpyAvx2Kernel::apply(int64_t n, float a, PCFp32 x, PFp32 y) {
  __m256 a00 = _mm256_broadcast_ss(&a);
  __m256 x00, y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);

    y00 = _mm256_fmadd_ps(a00, x00, y00);
    _mm256_storeu_ps(py, y00);

    px += 8;
    py += 8;
  }

  for (int i = 0; i < nr; ++i) {
    *py++ += a * *px++;
  }
}

void SAxpyAvx2Kernel::applyColumn(const SGEMVArgs &args, int column, float *y) {
  apply(args.N, args.x[column], args.A + column * args.lda, y);
}

float SDotAvx2Kernel::apply(int64_t n, const float *x, const float *y) {
  __m256 x00, y00, a00;

  a00 = _mm256_setzero_ps();

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  const float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);
    a00 = _mm256_fmadd_ps(x00, y00, a00);

    px += 8;
    py += 8;
  }

  // unroll a00
  __m128 r4 = _mm_add_ps(_mm256_extractf128_ps(a00, 1), _mm256_castps256_ps128(a00));
  __m128 r4h = _mm_movehl_ps(r4, r4);
  __m128 r2 = _mm_add_ps(r4, r4h);
  __m128 r2h = _mm_movehdup_ps(r2);
  __m128 r1 = _mm_add_ps(r2, r2h);
  float sum = _mm_cvtss_f32(r1);

  for (int i = 0; i < nr; ++i) {
    sum += *px++ * *py++;
  }

  return sum;
}

float SDotAvx2Kernel::applyRow(const SGEMVArgs &args, int row) {
  return apply(args.N, args.A + row * args.lda, args.x);
}

float DotQ4SymAvx2Kernel::apply(int64_t n, PCFp32 x, PCQ4x2 y, PCFp16 scaleY) {
  __m256 x00, y00, a00, ymmScale;
  __m256i yint8x32, yint8x32odd, yint8x32even, ymm0xf, ymm0x8;
  __m128i yint8x16;

  a00 = _mm256_setzero_ps();
  ymm0xf = _mm256_set1_epi8(0xf);
  ymm0x8 = _mm256_set1_epi8(0x8);
  
  int64_t nb = n / 32;

  PCFp32 px = x;
  PCQ4x2 py = y;
  for (int i = 0; i < nb; ++i) {
    float scale = _cvtsh_ss(*scaleY);
    ymmScale = _mm256_set1_ps(scale);
  
    // read 32 int4 (16 bytes), convert to 32 int8 and store to yint8x32 
    yint8x32 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(py)));
    yint8x32odd = _mm256_slli_epi16(yint8x32, 8);
    yint8x32even = _mm256_srli_epi16(yint8x32, 4);
    yint8x32 = _mm256_or_si256(yint8x32odd, yint8x32even);
    yint8x32 = _mm256_and_si256(yint8x32, ymm0xf);

    // uint4 range [0, 15] to int4 [-8, 7]
    yint8x32 = _mm256_sub_epi8(yint8x32, ymm0x8);

    // subblock 0
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(yint8x32, 0)));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 1
    yint8x16 = _mm256_extracti128_si256(yint8x32, 0);
    yint8x16 = _mm_srli_si128(yint8x16, 8);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 2
    yint8x16 = _mm256_extracti128_si256(yint8x32, 1);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 3
    yint8x16 = _mm_srli_si128(yint8x16, 8);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    py += 16;
    scaleY += 1;
  }

  // unroll a00
  __m128 r4 = _mm_add_ps(_mm256_extractf128_ps(a00, 1), _mm256_castps256_ps128(a00));
  __m128 r4h = _mm_movehl_ps(r4, r4);
  __m128 r2 = _mm_add_ps(r4, r4h);
  __m128 r2h = _mm_movehdup_ps(r2);
  __m128 r1 = _mm_add_ps(r2, r2h);
  float sum = _mm_cvtss_f32(r1);

  return sum;
}

float DotQ4Avx2Kernel::apply(int64_t n, PCFp32 x, PCQ4x2 y, PCFp16 scaleY, PCInt8 zpY) {
  __m256 x00, y00, a00, ymmScale;
  __m256i yint8x32, yint8x32odd, yint8x32even, ymm0xf, zeroPointv32;
  __m128i yint8x16;

  a00 = _mm256_setzero_ps();
  ymm0xf = _mm256_set1_epi8(0xf);
  
  int64_t nb = n / 32;

  PCFp32 px = x;
  PCQ4x2 py = y;
  PCInt8 pyzp = zpY;
  for (int i = 0; i < nb; ++i) {
    ymmScale = _mm256_set1_ps(_cvtsh_ss(*scaleY));
    zeroPointv32 = _mm256_set1_epi8(*pyzp);
  
    // read 32 int4 (16 bytes), convert to 32 int8 and store to yint8x32 
    yint8x32 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(py)));
    yint8x32odd = _mm256_slli_epi16(yint8x32, 8);
    yint8x32even = _mm256_srli_epi16(yint8x32, 4);
    yint8x32 = _mm256_or_si256(yint8x32odd, yint8x32even);
    yint8x32 = _mm256_and_si256(yint8x32, ymm0xf);

    yint8x32 = _mm256_sub_epi8(yint8x32, zeroPointv32);

    // subblock 0
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(yint8x32, 0)));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 1
    yint8x16 = _mm256_extracti128_si256(yint8x32, 0);
    yint8x16 = _mm_srli_si128(yint8x16, 8);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 2
    yint8x16 = _mm256_extracti128_si256(yint8x32, 1);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 3
    yint8x16 = _mm_srli_si128(yint8x16, 8);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    py += 16;
    scaleY += 1;
    pyzp += 1;
  }

  // unroll a00
  __m128 r4 = _mm_add_ps(_mm256_extractf128_ps(a00, 1), _mm256_castps256_ps128(a00));
  __m128 r4h = _mm_movehl_ps(r4, r4);
  __m128 r2 = _mm_add_ps(r4, r4h);
  __m128 r2h = _mm_movehdup_ps(r2);
  __m128 r1 = _mm_add_ps(r2, r2h);
  float sum = _mm_cvtss_f32(r1);

  return sum;
}

float DotQ4Avx2Kernel::applyRow(const Q4GemvArgs &args, int row) {
  PCQ4x2 data = args.A + row * args.N / 2;
  PCFp16 scale = args.scaleA + row * args.N / Q4GroupSize;
  PCInt8 zeroPoint = args.zeroPointA + row * args.N / Q4GroupSize;

  return apply(args.N, args.x, data, scale, zeroPoint);
}

float DotQ4SymAvx2Kernel::applyRow(const QGEMVInt4AArgs &args, int row) {
  PCQ4x2 data = args.A + row * args.N / 2;
  PCFp16 scale = args.scaleA + row * args.N / Q4GroupSize;

  return apply(args.N, args.x, data, scale);
}

void AxpyQ4SymAvx2Kernel::apply(
    int64_t n, float a, const uint8_t *x, const Fp16 *scaleX, float *y) {
  __m256 xv8, yv8, scalev8;
  __m256i xi8v32, xi8x32p0, xi8x32p1, v0xfv32, v0x8v32;
  __m128i xi8v16;

  __m256 av8 = _mm256_set1_ps(a);

  v0xfv32 = _mm256_set1_epi8(0xf);
  v0x8v32 = _mm256_set1_epi8(0x8);
  
  int64_t nb = n / 32;
  assert(n % 32 == 0);

  const uint8_t *px = x;
  const Fp16 *pxscale = scaleX;
  float *py = y;
  for (int i = 0; i < nb; ++i) {
    scalev8 = _mm256_set1_ps(_cvtsh_ss(*pxscale));

    // read 32 int4 (16 bytes), convert to 32 int8 and store to xi8x32.
    // Here is the steps of converting int4 to int8:
    // 
    // Input:
    // +---+---+
    // | A | B | <- packed 2 uint4 values A and B  into a byte
    // +---+---+
    // 
    // u8 -> i16
    // +---+---+---+---+
    // | 0 | 0 | A | B |
    // +---+---+---+---+
    //
    // i16 SHIFT-LEFT 8
    // +---+---+---+---+
    // | A | B | 0 | 0 |
    // +---+---+---+---+
    //
    // i16 SHIFT-RIGHT 4
    // +---+---+---+---+
    // | 0 | 0 | 0 | A |
    // +---+---+---+---+
    //
    // i16 (SHIFT-LEFT 8) OR (SHIFT-RIGHT 4)
    // +---+---+---+---+
    // | A | B | 0 | A |
    // +---+---+---+---+
    //
    // As 2 int8 (little-endian)
    // +---+---+  +---+---+
    // | 0 | A |  | A | B |
    // +---+---+  +---+---+
    //
    // AND 0xf
    // +---+---+  +---+---+
    // | 0 | A |  | 0 | B |
    // +---+---+  +---+---+
    xi8v32 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(px)));
    xi8v32 = _mm256_or_si256(_mm256_slli_epi16(xi8v32, 8), _mm256_srli_epi16(xi8v32, 4));
    xi8v32 = _mm256_and_si256(xi8v32, v0xfv32);

    // uint4 range [0, 15] to int4 [-8, 7]
    xi8v32 = _mm256_sub_epi8(xi8v32, v0x8v32);

    // vecror 8 subblock 0
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(xi8v32, 0)));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    // vecror 8 subblock 1
    xi8v16 = _mm256_extracti128_si256(xi8v32, 0);
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    // vecror 8 subblock 2
    xi8v16 = _mm256_extracti128_si256(xi8v32, 1);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    // vecror 8 subblock 3
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    px += 16;
    pxscale += 1;
  }
}

void AxpyQ4Avx2Kernel::apply(int64_t n, float a, PCQ4x2 x, PCFp16 scaleX, PCInt8 zpX, PFp32 y) {
  __m256 xv8, yv8, scalev8;
  __m256i xi8v32, xi8x32p0, xi8x32p1, v0xfv32, zeroPointv32;
  __m128i xi8v16;

  __m256 av8 = _mm256_set1_ps(a);

  v0xfv32 = _mm256_set1_epi8(0xf);
  
  int64_t nb = n / 32;
  assert(n % 32 == 0);

  PCQ4x2 px = x;
  PCFp16 pxscale = scaleX;
  PCInt8 pxzp = zpX;
  PFp32 py = y;
  for (int i = 0; i < nb; ++i) {
    scalev8 = _mm256_set1_ps(_cvtsh_ss(*pxscale));
    zeroPointv32 = _mm256_set1_epi8(*pxzp);

    xi8v32 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(px)));
    xi8v32 = _mm256_or_si256(_mm256_slli_epi16(xi8v32, 8), _mm256_srli_epi16(xi8v32, 4));
    xi8v32 = _mm256_and_si256(xi8v32, v0xfv32);

    // Q - zero_point
    xi8v32 = _mm256_sub_epi8(xi8v32, zeroPointv32);

    // vecror 8 subblock 0
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(xi8v32, 0)));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    // vecror 8 subblock 1
    xi8v16 = _mm256_extracti128_si256(xi8v32, 0);
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    // vecror 8 subblock 2
    xi8v16 = _mm256_extracti128_si256(xi8v32, 1);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    // vecror 8 subblock 3
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    yv8 = _mm256_loadu_ps(py);
    yv8 = _mm256_fmadd_ps(av8, xv8, yv8);
    _mm256_storeu_ps(py, yv8);
    py += 8;

    px += 16;
    pxscale += 1;
    pxzp += 1;
  }
}

void AxpyQ4Avx2Kernel::applyColumn(const Q4GemvArgs &args, int col, float *y) {
  PCQ4x2 data = args.A + col * args.N / 2;
  PCFp16 scale = args.scaleA + col * args.N / Q4GroupSize;
  PCInt8 zp = args.zeroPointA + col * args.N / Q4GroupSize;
  apply(args.N, args.x[col], data, scale, zp, y);
}

void DequantQ4SymAvx2Knl::apply(int n, PCQ4x2 src, PCFp16 scale, PFp32 tgt) {
  __m256 xv8, yv8, scalev8;
  __m256i xi8v32, xi8x32p0, xi8x32p1, v0xfv32, v0x8v32;
  __m128i xi8v16;

  v0xfv32 = _mm256_set1_epi8(0xf);
  v0x8v32 = _mm256_set1_epi8(0x8);
  
  int64_t nb = n / 32;
  assert(n % 32 == 0);

  PCQ4x2 px = src;
  PCFp16 pxscale = scale;
  PFp32 py = tgt;
  for (int i = 0; i < nb; ++i) {
    scalev8 = _mm256_set1_ps(_cvtsh_ss(*pxscale));

    xi8v32 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(px)));
    xi8v32 = _mm256_or_si256(_mm256_slli_epi16(xi8v32, 8), _mm256_srli_epi16(xi8v32, 4));
    xi8v32 = _mm256_and_si256(xi8v32, v0xfv32);

    // uint4 range [0, 15] to int4 [-8, 7]
    xi8v32 = _mm256_sub_epi8(xi8v32, v0x8v32);

    // vecror 8 subblock 0
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(xi8v32, 0)));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    // vecror 8 subblock 1
    xi8v16 = _mm256_extracti128_si256(xi8v32, 0);
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    // vecror 8 subblock 2
    xi8v16 = _mm256_extracti128_si256(xi8v32, 1);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    // vecror 8 subblock 3
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    px += 16;
    pxscale += 1;
  }
}

void DequantQ4Avx2Kernel::apply(int n, PCQ4x2 src, PCFp16 scale, PCInt8 zero, PFp32 tgt) {
  __m256 xv8, yv8, scalev8;
  __m256i xi8v32, xi8x32p0, xi8x32p1, v0xfv32, zeroPointv32;
  __m128i xi8v16;

  v0xfv32 = _mm256_set1_epi8(0xf);
  
  int64_t nb = n / 32;
  assert(n % 32 == 0);

  PCQ4x2 px = src;
  PCFp16 pxscale = scale;
  PFp32 py = tgt;
  PCInt8 pzp = zero;
  for (int i = 0; i < nb; ++i) {
    scalev8 = _mm256_set1_ps(_cvtsh_ss(*pxscale));
    zeroPointv32 = _mm256_set1_epi8(*pzp);

    xi8v32 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(px)));
    xi8v32 = _mm256_or_si256(_mm256_slli_epi16(xi8v32, 8), _mm256_srli_epi16(xi8v32, 4));
    xi8v32 = _mm256_and_si256(xi8v32, v0xfv32);

    // uint4 range [0, 15] to int4 [-8, 7]
    xi8v32 = _mm256_sub_epi8(xi8v32, zeroPointv32);

    // vecror 8 subblock 0
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(xi8v32, 0)));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    // vecror 8 subblock 1
    xi8v16 = _mm256_extracti128_si256(xi8v32, 0);
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    // vecror 8 subblock 2
    xi8v16 = _mm256_extracti128_si256(xi8v32, 1);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    // vecror 8 subblock 3
    xi8v16 = _mm_srli_si128(xi8v16, 8);
    xv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(xi8v16));
    xv8 = _mm256_mul_ps(xv8, scalev8);
    _mm256_storeu_ps(py, xv8);
    py += 8;

    px += 16;
    pxscale += 1;
    pzp += 1;
  }
}

void AxpyQ4SymAvx2Kernel::applyColumn(const QGEMVInt4AArgs &args, int col, float *y) {
  const uint8_t *data = args.A + col * args.N / 2;
  const Fp16 *scale = args.scaleA + col * args.N / 32;
  apply(args.N, args.x[col], data, scale, y);
}

// real_value = A * quantized_value + B
void DequantInt8BAvx2Kernel::apply(
    int64_t n, const uint8_t *data, const float *scaleZp, int64_t offset, float *tgt) {
  int64_t hBegin = offset % Int8bScaleGroupSize;
  int64_t tEnd = (offset + n - 1) % Int8bScaleGroupSize + 1;
  int64_t nb = (n + Int8bScaleGroupSize - 1) / Int8bScaleGroupSize + (hBegin < tEnd ? 0 : 1);

  const float *pscaleZp = scaleZp;
  const uint8_t *pdata = data + offset;
  float *ptgt = tgt;
  for (int i = 0; i < nb; ++i) {
    float scale = *pscaleZp++;
    float zeroPoint = *pscaleZp++;

    __m256 ymmScale = _mm256_broadcast_ss(&scale);
    __m256 ymmZeroPoint = _mm256_broadcast_ss(&zeroPoint);

    int64_t begin = i == 0 ? hBegin : 0;
    int64_t end = i == nb - 1 ? tEnd : Int8bScaleGroupSize;

    assert(end - begin > 0);
    int64_t nb2 = (end - begin) / 16;
    int64_t nr2 = (end - begin) % 16;
    for (int j = 0; j < nb2; ++j) {
      __m128i xmmQx16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pdata));
      __m256 ymmRx8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(xmmQx16));
      ymmRx8 = _mm256_fmadd_ps(ymmRx8, ymmScale, ymmZeroPoint);
      _mm256_storeu_ps(ptgt, ymmRx8);
      ptgt += 8;

      xmmQx16 = _mm_srli_si128(xmmQx16, 8);
      ymmRx8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(xmmQx16));
      ymmRx8 = _mm256_fmadd_ps(ymmRx8, ymmScale, ymmZeroPoint);
      _mm256_storeu_ps(ptgt, ymmRx8);
      ptgt += 8;

      pdata += 16;
    }

    for (int j = 0; j < nr2; ++j) {
      *ptgt++ = scale * (*pdata++) + zeroPoint;
    }
  }
}


}  // namespace lymath
