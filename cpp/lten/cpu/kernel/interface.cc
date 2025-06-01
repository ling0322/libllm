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

#include "lten/cpu/kernel/interface.h"

#include <stdlib.h>

#include <atomic>
#include <memory>

#include "lten/cpu/kernel/abstract.h"
#include "lten/cpu/kernel/asimdhp.h"
#include "lten/cpu/kernel/avx2.h"
#include "lten/cpu/kernel/avx512.h"
#include "lten/cpu/kernel/cvt.h"
#include "lten/cpu/kernel/fallback.h"
#include "lten/cpu/kernel/gemm.h"
#include "lten/cpu/kernel/gemv.h"
#include "lten/cpu/kernel/interface.h"
#include "lutil/is_debug.h"
#include "lutil/log.h"
#include "lutil/platform.h"
#include "lutil/strings.h"
#include "ruapu/ruapu.h"

namespace lten {
namespace op {
namespace cpu {
namespace kernel {

CpuMathBackend findDefaultCpuMathBackend() {
#if LUT_CPU_ARCH == LUT_AMD64
  return CpuMathBackend::AVX2;
#elif LUT_CPU_ARCH == LUT_AARCH64
  return CpuMathBackend::ASIMDHP;
#else
  NOT_IMPL();
#endif
}

CpuMathBackend findBestCpuMathBackend() {
  if (lut::isDebug()) return findDefaultCpuMathBackend();

  ruapu_init();

#if LUT_CPU_ARCH == LUT_AMD64
  bool isaAvx2 = ruapu_supports("avx2") > 0;
  bool isaAvx512f = ruapu_supports("avx512f") > 0;
  bool isaF16c = ruapu_supports("f16c") > 0;

  LOG(INFO)
      << lut::sprintf("ISA support: AVX2=%d F16C=%d AVX512F=%d", isaAvx2, isaF16c, isaAvx512f);

  if (isaAvx512f && isaF16c) {
    LOG(INFO) << "Use Avx512 backend.";
    return CpuMathBackend::AVX512;
  }

  if (isaAvx2 && isaF16c) {
    LOG(INFO) << "Use Avx2 backend.";
    return CpuMathBackend::AVX2;
  }
#elif LUT_CPU_ARCH == LUT_AARCH64
  LOG(INFO) << "Use asimdhp backend.";
  return CpuMathBackend::ASIMDHP;
#else

  LOG(FATAL) << "CPU not supported.";
  NOT_IMPL();
#endif
  LOG(INFO) << "Fallback to the slow CPU kernels since no Avx2 or Asimdhp support.";
  return CpuMathBackend::FALLBACK;
}

bool gAllowSlowKernel = false;
CpuMathBackend gDefaultBackend = CpuMathBackend::UNKNOWN;

CpuMathBackend getCpuMathBackend(CpuMathBackend fromBackend) {
  if (fromBackend == CpuMathBackend::DEFAULT) {
    CHECK(gDefaultBackend != CpuMathBackend::UNKNOWN) << "math kernel not initialized";
    return gDefaultBackend;
  } else {
    return fromBackend;
  }
}

void init() {
  gDefaultBackend = findBestCpuMathBackend();
}

void destroy() {
}

void setAllowSlowKernel(bool allow) {
  gAllowSlowKernel = allow;
}

void gemmFloat(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType) {
  GemmArgs<float, float, float> args;
  args.transA = transA;
  args.transB = transB;
  args.M = M;
  args.N = N;
  args.K = K;
  args.A = A;
  args.lda = lda;
  args.B = B;
  args.ldb = ldb;
  args.C = C;
  args.ldc = ldc;

  backendType = getCpuMathBackend(backendType);
  if (false) {
#if LUT_CPU_ARCH == LUT_AMD64
  } else if (backendType == CpuMathBackend::AVX2 && mode == Mode::OMP) {
    gemm<288, 512, 4096, 6, 16, float, CpuMathBackend::AVX2, Mode::OMP>(args);
  } else if (backendType == CpuMathBackend::AVX2 && mode == Mode::SingleThread) {
    gemm<288, 512, 4096, 6, 16, float, CpuMathBackend::AVX2, Mode::SingleThread>(args);
  } else if (backendType == CpuMathBackend::AVX512 && mode == Mode::OMP) {
    gemm<576, 512, 4096, 12, 32, float, CpuMathBackend::AVX512, Mode::OMP>(args);
  } else if (backendType == CpuMathBackend::AVX512 && mode == Mode::SingleThread) {
    gemm<576, 512, 4096, 12, 32, float, CpuMathBackend::AVX512, Mode::SingleThread>(args);
#elif LUT_CPU_ARCH == LUT_AARCH64
  } else if (gAllowSlowKernel && backendType == CpuMathBackend::ASIMDHP && mode == Mode::OMP) {
    gemm<288, 512, 4096, 6, 16, float, CpuMathBackend::FALLBACK, Mode::OMP>(args);
  } else if (
      gAllowSlowKernel && backendType == CpuMathBackend::ASIMDHP && mode == Mode::SingleThread) {
    gemm<288, 512, 4096, 6, 16, float, CpuMathBackend::FALLBACK, Mode::OMP>(args);
#endif
  } else {
    NOT_IMPL();
  }
}

void gemmHalf(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const Float16 *B,
    int ldb,
    Float16 *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType) {
  GemmArgs<Float16, Float16, Float16> args;
  args.transA = transA;
  args.transB = transB;
  args.M = M;
  args.N = N;
  args.K = K;
  args.A = A;
  args.lda = lda;
  args.B = B;
  args.ldb = ldb;
  args.C = C;
  args.ldc = ldc;

  backendType = getCpuMathBackend(backendType);
  if (false) {
#if LUT_CPU_ARCH == LUT_AARCH64
  } else if (backendType == CpuMathBackend::ASIMDHP && mode == Mode::OMP) {
    gemm<576, 512, 4096, 12, 16, Float16, CpuMathBackend::ASIMDHP, Mode::OMP>(args);
  } else if (backendType == CpuMathBackend::ASIMDHP && mode == Mode::SingleThread) {
    gemm<576, 512, 4096, 12, 16, Float16, CpuMathBackend::ASIMDHP, Mode::SingleThread>(args);
#endif
  } else {
    NOT_IMPL();
  }
}

void dequantQInt4ToFloat(
    int n,
    const QInt4x32 *data,
    int offset,
    float *tgt,
    Mode mode,
    CpuMathBackend backendType) {
  backendType = getCpuMathBackend(backendType);

  if (false) {
#if LUT_CPU_ARCH == LUT_AMD64
  } else if (backendType == CpuMathBackend::AVX2 && mode == Mode::OMP) {
    cvt<QInt4x32, float, CpuMathBackend::AVX2, Mode::OMP>(n, data, offset, tgt, 0);
  } else if (backendType == CpuMathBackend::AVX512 && mode == Mode::OMP) {
    cvt<QInt4x32, float, CpuMathBackend::AVX2, Mode::OMP>(n, data, offset, tgt, 0);
#endif
  } else {
    NOT_IMPL();
  }
}

void quantFloatToQInt4(
    int n,
    const float *data,
    int offset,
    QInt4x32 *tgt,
    Mode mode,
    CpuMathBackend backendType) {
  if (mode == Mode::OMP) {
    cvt<float, QInt4x32, CpuMathBackend::FALLBACK, Mode::OMP>(n, data, offset, tgt, 0);
  } else if (mode == Mode::SingleThread) {
    cvt<float, QInt4x32, CpuMathBackend::FALLBACK, Mode::SingleThread>(n, data, offset, tgt, 0);
  } else {
    NOT_IMPL();
  }
}

void dequantQInt4ToHalf(
    int n,
    const QInt4x32 *data,
    int offset,
    Float16 *tgt,
    Mode mode,
    CpuMathBackend backendType) {
  backendType = getCpuMathBackend(backendType);

  if (false) {
#if LUT_CPU_ARCH == LUT_AARCH64
  } else if (backendType == CpuMathBackend::ASIMDHP && mode == Mode::OMP) {
    cvt<QInt4x32, Float16, CpuMathBackend::ASIMDHP, Mode::OMP>(n, data, offset, tgt, 0);
#endif
  } else {
    NOT_IMPL();
  }
}

void gemmFloatQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const QInt4x32 *B,
    float *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType) {
  GemmArgs<float, QInt4x32, float> args;
  args.transA = transA;
  args.transB = transB;
  args.M = M;
  args.N = N;
  args.K = K;
  args.A = A;
  args.lda = lda;
  args.B = B;
  args.ldb = transB ? K : N;
  args.C = C;
  args.ldc = ldc;

  backendType = getCpuMathBackend(backendType);

  if (false) {
#if LUT_CPU_ARCH == LUT_AMD64
  } else if (backendType == CpuMathBackend::AVX2 && mode == Mode::OMP) {
    qgemm<288, 512, 4096, 6, 16, float, QInt4x32, CpuMathBackend::AVX2, Mode::OMP>(args);
  } else if (backendType == CpuMathBackend::AVX2 && mode == Mode::SingleThread) {
    qgemm<288, 512, 4096, 6, 16, float, QInt4x32, CpuMathBackend::AVX2, Mode::SingleThread>(args);
  } else if (backendType == CpuMathBackend::AVX512 && mode == Mode::OMP) {
    qgemm<576, 512, 4096, 12, 32, float, QInt4x32, CpuMathBackend::AVX512, Mode::OMP>(args);
  } else if (backendType == CpuMathBackend::AVX512 && mode == Mode::SingleThread) {
    qgemm<576, 512, 4096, 12, 32, float, QInt4x32, CpuMathBackend::AVX512, Mode::SingleThread>(
        args);
#elif LUT_CPU_ARCH == LUT_AARCH64
  } else if (gAllowSlowKernel && backendType == CpuMathBackend::ASIMDHP && mode == Mode::OMP) {
    qgemm<288, 512, 4096, 6, 16, float, QInt4x32, CpuMathBackend::FALLBACK, Mode::OMP>(args);
  } else if (
      gAllowSlowKernel && backendType == CpuMathBackend::ASIMDHP && mode == Mode::SingleThread) {
    qgemm<288, 512, 4096, 6, 16, float, QInt4x32, CpuMathBackend::FALLBACK, Mode::SingleThread>(
        args);
#endif
  } else {
    NOT_IMPL();
  }
}

void gemmHalfQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const QInt4x32 *B,
    Float16 *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType) {
  GemmArgs<Float16, QInt4x32, Float16> args;
  args.transA = transA;
  args.transB = transB;
  args.M = M;
  args.N = N;
  args.K = K;
  args.A = A;
  args.lda = lda;
  args.B = B;
  args.ldb = transB ? K : N;
  args.C = C;
  args.ldc = ldc;

  backendType = getCpuMathBackend(backendType);

  if (false) {
#if LUT_CPU_ARCH == LUT_AARCH64
  } else if (backendType == CpuMathBackend::ASIMDHP && mode == Mode::OMP) {
    qgemm<576, 512, 4096, 12, 16, Float16, QInt4x32, CpuMathBackend::ASIMDHP, Mode::OMP>(args);
  } else if (backendType == CpuMathBackend::ASIMDHP && mode == Mode::SingleThread) {
    qgemm<576, 512, 4096, 12, 16, Float16, QInt4x32, CpuMathBackend::ASIMDHP, Mode::SingleThread>(
        args);
#endif
  } else {
    NOT_IMPL();
  }
}

void convertHalfToFloat(int n, const Float16 *x, float *y, Mode mode, CpuMathBackend backendType) {
  backendType = getCpuMathBackend(backendType);

  if (false) {
#if LUT_CPU_ARCH == LUT_AARCH64
  } else if (backendType == CpuMathBackend::ASIMDHP && mode == Mode::OMP) {
    cvt<Float16, float, CpuMathBackend::ASIMDHP, Mode::OMP>(n, x, 0, y, 0);
#elif LUT_CPU_ARCH == LUT_AMD64
  } else if (backendType == CpuMathBackend::AVX2 && mode == Mode::OMP) {
    cvt<Float16, float, CpuMathBackend::AVX2, Mode::OMP>(n, x, 0, y, 0);
  } else if (backendType == CpuMathBackend::AVX512 && mode == Mode::OMP) {
    cvt<Float16, float, CpuMathBackend::AVX512, Mode::OMP>(n, x, 0, y, 0);
#endif
  } else {
    NOT_IMPL();
  }
}

void convertFloatToHalf(int n, const float *x, Float16 *y, Mode mode, CpuMathBackend backendType) {
  backendType = getCpuMathBackend(backendType);

  if (false) {
#if LUT_CPU_ARCH == LUT_AARCH64
  } else if (backendType == CpuMathBackend::ASIMDHP && mode == Mode::OMP) {
    cvt<float, Float16, CpuMathBackend::ASIMDHP, Mode::OMP>(n, x, 0, y, 0);
#elif LUT_CPU_ARCH == LUT_AMD64
  } else if (backendType == CpuMathBackend::AVX2 && mode == Mode::OMP) {
    cvt<float, Float16, CpuMathBackend::FALLBACK, Mode::OMP>(n, x, 0, y, 0);
  } else if (backendType == CpuMathBackend::AVX512 && mode == Mode::OMP) {
    cvt<float, Float16, CpuMathBackend::FALLBACK, Mode::OMP>(n, x, 0, y, 0);
#endif
  } else {
    NOT_IMPL();
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace lten
