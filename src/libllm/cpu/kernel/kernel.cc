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
#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/cpu/kernel/cvt_h.h"
#include "libllm/cpu/kernel/dequant_sq4.h"
#include "libllm/cpu/kernel/gemm_h.h"
#include "libllm/cpu/kernel/gemm_hq4.h"
#include "libllm/cpu/kernel/gemm_sq4.h"
#include "libllm/cpu/kernel/gemm_s.h"
#include "ruapu/ruapu.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

enum class CPUMathBackend {
  DEFAULT,
  AVX2,
  AVX512,
  ASIMDHP
};

CPUMathBackend findBestCpuMathBackend() {
  ruapu_init();

#ifdef LUT_ARCH_AMD64
  bool isaAvx2 = ruapu_supports("avx2") > 0;
  bool isaAvx512f = ruapu_supports("avx512f") > 0;
  bool isaF16c = ruapu_supports("f16c") > 0;

  LOG(INFO) << lut::sprintf(
      "ISA support: AVX2=%d F16C=%d AVX512F=%d", isaAvx2, isaF16c, isaAvx512f);
  
  if (isaAvx512f && isaF16c) {
    LOG(INFO) << "Use Avx512 backend.";
    return CPUMathBackend::AVX512;
  }
  
  if (isaAvx2 && isaF16c) {
    LOG(INFO) << "Use Avx2 backend.";
    return CPUMathBackend::AVX2;
  }
#endif  // LUT_ARCH_AMD64

#ifdef LUT_ARCH_AARCH64
  LOG(INFO) << "Use asimdhp backend.";
  return CPUMathBackend::ASIMDHP;
#endif  // LUT_ARCH_AARCH64

  LOG(FATAL) << "CPU not supported.";
  NOT_IMPL();
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
  const Gemm<float> *getSgemm() const { return _sgemm.get(); }
  const Gemm<float> *getSgemmOmp() const { return _sgemmOmp.get(); }
  const Gemm<Float16> *getHgemm() const { return _hgemm.get(); }
  const Gemm<Float16> *getHgemmOmp() const { return _hgemmOmp.get(); }
  const QInt4Gemm<float> *getGemmQ4() const { return _q4gemm.get(); }
  const QInt4Gemm<Float16> *getHQInt4Gemm() const { return _hq4gemmParallel.get(); }
  const DequantQInt4<float> *getDequantQInt4() const { return _q4dequant.get(); }
  const DequantQInt4<Float16> *getDequantQInt4ToHalf() const { return _q4dequantHalf.get(); }
  const CvtHalf *getCvtHalf() const { return _cvtHalf.get(); }

 private:
  static Api *_instance;
  static int _numThreads;

  std::unique_ptr<Gemm<float>> _sgemm;
  std::unique_ptr<Gemm<float>> _sgemmOmp;
  std::unique_ptr<Gemm<Float16>> _hgemmOmp;
  std::unique_ptr<Gemm<Float16>> _hgemm;
  std::unique_ptr<QInt4Gemm<float>> _q4gemm;
  std::unique_ptr<QInt4Gemm<Float16>> _hq4gemmParallel;
  std::unique_ptr<DequantQInt4<float>> _q4dequant;
  std::unique_ptr<DequantQInt4<Float16>> _q4dequantHalf;
  std::unique_ptr<CvtHalf> _cvtHalf;
};
    

Api *Api::_instance = nullptr;
int Api::_numThreads = 1;

void Api::init() {
  if (_instance) {
    destroy();
  }

  _instance = new Api();
  switch (findBestCpuMathBackend()) {
#ifdef LUT_ARCH_AMD64
    case CPUMathBackend::AVX512:
      _instance->_sgemm = std::make_unique<SGEMMImplAvx512>();
      _instance->_sgemmOmp = std::make_unique<SGEMMImplAvx512OMP>();
      _instance->_q4gemm = std::make_unique<Q4GemmAvx512OMP>();
      _instance->_q4dequant = std::make_unique<DequantQInt4Avx2OMP>();
      _instance->_cvtHalf = std::make_unique<CvtHalfAvx2OMP>();
      break;
    case CPUMathBackend::AVX2:
      _instance->_sgemm = std::make_unique<SGEMMImplAvx2>();
      _instance->_sgemmOmp = std::make_unique<SGEMMImplAvx2OMP>();
      _instance->_q4gemm = std::make_unique<Q4GemmAvx2OMP>();
      _instance->_q4dequant = std::make_unique<DequantQInt4Avx2OMP>();
      _instance->_cvtHalf = std::make_unique<CvtHalfAvx2OMP>();
      break;
#endif  // LUT_ARCH_AMD64
#ifdef LUT_ARCH_AARCH64
    case CPUMathBackend::ASIMDHP:
      _instance->_hgemm = std::make_unique<HGemmAsimdhp>();
      _instance->_hgemmOmp = std::make_unique<HGemmAsimdhpOMP>();
      _instance->_sgemm = std::make_unique<SGEMMImplDefault>();
      _instance->_sgemmOmp = std::make_unique<SGEMMImplDefaultOMP>();
      _instance->_q4gemm = std::make_unique<Q4GemmFallbackOMP>();
      _instance->_hq4gemmParallel = std::make_unique<HQInt4GemmAsimdhpOMP>();
      _instance->_q4dequant = std::make_unique<DequantQInt4FallbackOMP>();
      _instance->_q4dequantHalf = std::make_unique<DequantQInt4ToHalfAsimdhpOMP>();
      _instance->_cvtHalf = std::make_unique<CvtHalfAsimdhpOMP>();
      break;
#endif  // LUT_ARCH_AARCH64
    case CPUMathBackend::DEFAULT:
      _instance->_sgemm = std::make_unique<SGEMMImplDefault>();
      _instance->_sgemmOmp = std::make_unique<SGEMMImplDefaultOMP>();
      _instance->_q4gemm = std::make_unique<Q4GemmFallbackOMP>();
      _instance->_q4dequant = std::make_unique<DequantQInt4FallbackOMP>();
      _instance->_cvtHalf = std::make_unique<CvtHalfFallbackOMP>();

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
    Mode mode) {
  GemmArgs<float> args;
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

  switch (mode) {
    case Mode::Auto:
    case Mode::OMP:
      Api::getInstance()->getSgemmOmp()->apply(args);
      break;
    case Mode::SingleThread:
      Api::getInstance()->getSgemm()->apply(args);
      break;
    default:
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
    Mode mode) {
  GemmArgs<Float16> args;
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

  switch (mode) {
    case Mode::Auto:
    case Mode::OMP:
      Api::getInstance()->getHgemmOmp()->apply(args);
      break;
    case Mode::SingleThread:
      Api::getInstance()->getHgemm()->apply(args);
      break;
    default:
      NOT_IMPL();
  }
}

void dequantQInt4ToFloat(
    int n,
    const UInt4x2 *data,
    const Float16 *scale,
    const UInt4x2 *zeroPoint,
    int offset,
    float *tgt,
    Mode mode) {
  DataQInt4 x(data, scale, zeroPoint);

  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getDequantQInt4()->apply(
          n,
          x,
          offset,
          tgt);
      break;
    default:
      NOT_IMPL();
  }
}

void dequantQInt4ToHalf(
    int n,
    const UInt4x2 *data,
    const Float16 *scale,
    const UInt4x2 *zeroPoint,
    int offset,
    Float16 *tgt,
    Mode mode) {
  DataQInt4 x(data, scale, zeroPoint);

  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getDequantQInt4ToHalf()->apply(
          n,
          x,
          offset,
          tgt);
      break;
    default:
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
    const UInt4x2 *dataB,
    const Float16 *scaleB,
    const UInt4x2 *zeroPointB,
    float *C,
    int ldc,
    Mode mode) {
  DataQInt4 B(dataB, scaleB, zeroPointB);

  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getGemmQ4()->apply(QInt4GemmArgs<float>{
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

void gemmHalfQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const UInt4x2 *dataB,
    const Float16 *scaleB,
    const UInt4x2 *zeroPointB,
    Float16 *C,
    int ldc,
    Mode mode) {
  DataQInt4 B(dataB, scaleB, zeroPointB);

  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getHQInt4Gemm()->apply(QInt4GemmArgs<Float16>{
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

void convertHalfToFloat(int n, const Float16 *x, float *y, Mode mode) {
  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getCvtHalf()->cvtHalfToFloat(n, x, y);
      break;
    default:
      NOT_IMPL();
  }
}

void convertFloatToHalf(int n, const float *x, Float16 *y, Mode mode) {
  switch (mode) {
    case Mode::Auto:
      Api::getInstance()->getCvtHalf()->cvtFloatToHalf(n, x, y);
      break;
    default:
      NOT_IMPL();
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
