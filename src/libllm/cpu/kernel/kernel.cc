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

#include "libllm/cpu/kernel/kernel.h"

#include <omp.h>
#include <stdlib.h>
#include <memory>
#include "libllm/lut/log.h"
#include "libllm/lut/platform.h"
#include "libllm/cpu/kernel/args.h"
#include "libllm/cpu/kernel/hcvt.h"
#include "libllm/cpu/kernel/q4dequant.h"
#include "libllm/cpu/kernel/q4gemm.h"
#include "libllm/cpu/kernel/sgemm.h"

namespace lymath {

enum class CPUMathBackend {
  DEFAULT,
  AVX2,
  AVX512
};

CPUMathBackend findBestCpuMathBackend() {
  if (lut::isAvx512Available()) {
    LOG(INFO) << "lymath: Use Avx512 backend.";
    return CPUMathBackend::AVX512;
  } else if (lut::isAvx2Available()) {
    LOG(INFO) << "lymath: Use Avx2 backend.";
    return CPUMathBackend::AVX2;
  } else {
    LOG(WARN) << "lymath: fallback to default backend.";
    return CPUMathBackend::AVX2;
  }
}

// instance of Api.
class Api;
static Api *gApiInstance = nullptr;

// implementation for lymath api.
class Api {
 public:
  static void init();
  static void destroy();
  static void setNumThreads(int numThreads);
  static int getNumThreads() { return _numThreads; }
  static const Api *getInstance();

  // get kernel implementations.
  const SGEMM *getSgemm() const { return _sgemm.get(); }
  const SGEMM *getSgemmOmp() const { return _sgemmOmp.get(); }
  const Q4Gemm *getQ4Gemm() const { return _q4gemm.get(); }
  const DequantQ4 *getDequantQ4() const { return _q4dequant.get(); }
  const CvtHalfToFloat *getCvtHalfToFloat() const { return _cvtHalfToFloat.get(); }

 private:
  static Api *_instance;
  static int _numThreads;

  std::unique_ptr<SGEMM> _sgemm;
  std::unique_ptr<SGEMM> _sgemmOmp;
  std::unique_ptr<Q4Gemm> _q4gemm;
  std::unique_ptr<DequantQ4> _q4dequant;
  std::unique_ptr<CvtHalfToFloat> _cvtHalfToFloat;
};

Api *Api::_instance = nullptr;
int Api::_numThreads = 1;

void Api::init() {
  if (_instance) {
    destroy();
  }

  _instance = new Api();
  switch (findBestCpuMathBackend()) {
    case CPUMathBackend::AVX512:
      _instance->_sgemm = std::make_unique<SGEMMImplAvx512>();
      _instance->_sgemmOmp = std::make_unique<SGEMMImplAvx512OMP>();
      _instance->_q4gemm = std::make_unique<Q4GemmAvx512OMP>();
      _instance->_q4dequant = std::make_unique<DequantQ4Avx2OMP>();
      _instance->_cvtHalfToFloat = std::make_unique<CvtHalfToFloatAvx2OMP>();
      break;
    case CPUMathBackend::AVX2:
      _instance->_sgemm = std::make_unique<SGEMMImplAvx2>();
      _instance->_sgemmOmp = std::make_unique<SGEMMImplAvx2OMP>();
      _instance->_q4gemm = std::make_unique<Q4GemmAvx2OMP>();
      _instance->_q4dequant = std::make_unique<DequantQ4Avx2OMP>();
      _instance->_cvtHalfToFloat = std::make_unique<CvtHalfToFloatAvx2OMP>();
      break;
    case CPUMathBackend::DEFAULT:
      _instance->_sgemm = std::make_unique<SGEMMImplDefault>();
      _instance->_sgemmOmp = std::make_unique<SGEMMImplDefaultOMP>();
      _instance->_q4gemm = std::make_unique<Q4GemmFallbackOMP>();
      _instance->_q4dequant = std::make_unique<DequantQ4FallbackOMP>();
      _instance->_cvtHalfToFloat = std::make_unique<CvtHalfToFloatFallbackOMP>();
      break;
    default:
      NOT_IMPL();
  }
}

void Api::destroy() {
  delete _instance;
  _instance = nullptr;
}

const Api *Api::getInstance() {
  CHECK(_instance);
  return _instance;
}

}  // namespace libllmmath

using lymath::Q4GemmArgs;
using lymath::Api;

void lymath_init() {
  Api::init();
}

void lymath_destroy() {
  Api::destroy();
}

void lymath_sgemm(
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
    int ldc) {
  Api::getInstance()->getSgemm()->apply(
      transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
}

void lymath_sgemm_omp(
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
    int ldc) {
  Api::getInstance()->getSgemmOmp()->apply(
      transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
}

void lymath_dequant_q4(
    int n,
    const lymath_q4x2_t *data,
    const lymath_float16_t *scale,
    const uint8_t *zeroPoint,
    int offset,
    float *tgt) {
  lymath::DataQ4 x(
    reinterpret_cast<lymath::PCQ4x2>(data),
    reinterpret_cast<lymath::PCFp16>(scale),
    reinterpret_cast<lymath::PCUInt8>(zeroPoint));

  Api::getInstance()->getDequantQ4()->apply(
      n,
      x,
      offset,
      tgt);
}

void lymath_q4gemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const lymath_q4x2_t *dataB,
    const lymath_float16_t *scaleB,
    const uint8_t *zeroPointB,
    float *C,
    int ldc) {
  lymath::DataQ4 B(
      reinterpret_cast<lymath::PCQ4x2>(dataB),
      reinterpret_cast<lymath::PCFp16>(scaleB),
      reinterpret_cast<lymath::PCUInt8>(zeroPointB));
  Api::getInstance()->getQ4Gemm()->apply(Q4GemmArgs{
      transA,
      transB,
      M,
      N,
      K,
      A,
      lda,
      B,
      C,
      ldc});
}

void lymath_half2float(int n, const lymath_float16_t *x, float *y) {
  Api::getInstance()->getCvtHalfToFloat()->apply(n, x, y);
}
