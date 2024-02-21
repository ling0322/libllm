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
#include "libllm/lut/strings.h"
#include "libllm/cpu/kernel/args.h"
#include "libllm/cpu/kernel/hcvt.h"
#include "libllm/cpu/kernel/q4dequant.h"
#include "libllm/cpu/kernel/q4gemm.h"
#include "libllm/cpu/kernel/sgemm.h"
#include "ruapu/ruapu.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

enum class CPUMathBackend {
  DEFAULT,
  AVX2,
  AVX512
};

CPUMathBackend findBestCpuMathBackend() {
  raupu_init();
  bool isaAvx2 = ruapu_supports("avx2") > 0;
  bool isaAvx512f = ruapu_supports("avx512f") > 0;
  bool isaF16c = ruapu_supports("f16c") > 0;

  LOG(INFO) << lut::sprintf(
      "ISA support: AVX2=%d F16C=%d AVX512F=%d", isaAvx2, isaF16c, isaAvx512f);

  if (isaAvx512f && isaF16c) {
    LOG(INFO) << "Use Avx512 backend.";
    return CPUMathBackend::AVX512;
  } else if (isaAvx2 && isaF16c) {
    LOG(INFO) << "Use Avx2 backend.";
    return CPUMathBackend::AVX2;
  } else {
    LOG(FATAL) << "CPU not supported (AVX2 and F16C is required).";
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

// -----------------------------------------------------------------------------------------------+
// API implementation                                                                             |
// -----------------------------------------------------------------------------------------------+

void init() {
  Api::init();
}

void destroy() {
  Api::destroy();
}

void sgemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Fp32 *A,
    int lda,
    const Fp32 *B,
    int ldb,
    Fp32 *C,
    int ldc,
    Mode mode) {
  switch (mode) {
    case Mode::Auto:
    case Mode::OMP:
      Api::getInstance()->getSgemmOmp()->apply(
          transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
      break;
    case Mode::SingleThread:
      Api::getInstance()->getSgemm()->apply(
          transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
      break;
    default:
      NOT_IMPL();
  }
}

void dequantQ4(
    int n,
    const Q4x2 *data,
    const Fp16 *scale,
    const UInt8 *zeroPoint,
    int offset,
    float *tgt,
    Mode mode) {
  DataQ4 x(data, scale, zeroPoint);

  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getDequantQ4()->apply(
          n,
          x,
          offset,
          tgt);
      break;
    default:
      NOT_IMPL();
  }
}

void gemmQ4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const Q4x2 *dataB,
    const Fp16 *scaleB,
    const UInt8 *zeroPointB,
    float *C,
    int ldc,
    Mode mode) {
  DataQ4 B(dataB, scaleB, zeroPointB);

  switch (mode) {
    case Mode::Auto:
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
      break;
    default:
      NOT_IMPL();
  }
}

void convertHalfToFloat(int n, const Fp16 *x, float *y, Mode mode) {
  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getCvtHalfToFloat()->apply(n, x, y);
      break;
    default:
      NOT_IMPL();
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
