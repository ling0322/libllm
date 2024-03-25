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

#pragma once

#include <stdint.h>
#include "libllm/cpu/kernel/kernel.h"

#define LIBLLM_KERNEL_MSVC (_MSC_VER && !__INTEL_COMPILER)

#if LIBLLM_KERNEL_MSVC
#define LIBLLM_KERNEL_FORCE_INLINE __forceinline
#else
#define LIBLLM_KERNEL_FORCE_INLINE __attribute__((always_inline)) inline
#endif

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

constexpr int GEMVMinRowsPerThread = 128;
constexpr int CvtMinElemPerThread = 1024;
constexpr int DequantMinElemPerThread = 1024;
constexpr int GroupSizeQInt4 = 128;

class DataQInt4 {
 public:
  constexpr DataQInt4(): 
      _data(nullptr),
      _scale(nullptr),
      _zero(nullptr) {}
  constexpr DataQInt4(const UInt4x2 *data, const Float16 *scale, const UInt4x2 *zero) :
      _data(data),
      _scale(scale),
      _zero(zero) {}

  constexpr const UInt4x2 *getDataByGroup(int64_t groupIdx) const {
    return _data + groupIdx * GroupSizeQInt4 / 2;
  }
  constexpr const Float16 *getScaleByGroup(int64_t groupIdx) const { return _scale + groupIdx; }
  constexpr const UInt4x2 *getZeroByGroup(int64_t groupIdx) const { return _zero + groupIdx / 2; }

  constexpr Float16 getScaleValByGroup(int64_t groupIdx) const { return _scale[groupIdx]; }
  constexpr uint8_t getZeroValByGroup(int64_t groupIdx) const {
    return (_zero[groupIdx >> 1].b >> ((groupIdx & 1) << 2)) & 0xf;
  }

 private:
  const UInt4x2 *_data;
  const Float16 *_scale;
  const UInt4x2 *_zero;
};

template<typename T>
struct GemvArgs {
  typedef T VecType;

  bool transA;
  int M;
  int N;
  const T *A;
  int lda;
  const T *x;
  int incX;
  T *y;
  int incY;
};

typedef GemvArgs<float> SGEMVArgs;
typedef GemvArgs<Float16> HGemvArgs;

struct Q4GemvArgs {
  typedef float VecType;

  bool transA;
  int M;
  int N;
  DataQInt4 A;
  const float *x;
  int incX;
  float *y;
  int incY;
};

template<typename T>
struct GemmArgs {
  bool transA;
  bool transB;
  int M;
  int N;
  int K;
  const T *A;
  int lda;
  const T *B;
  int ldb;
  T *C;
  int ldc;
};

struct GemmQ4Args {
  bool transA;
  bool transB;
  int M;
  int N;
  int K;
  const float *A;
  int lda;
  DataQInt4 B;
  float *C;
  int ldc;
};

template<typename T>
class DequantQInt4 {
 public:
  virtual ~DequantQInt4() = default;
  virtual void apply(int n, DataQInt4 x, int64_t offsetX, T *y) const = 0;
};

template<typename T>
class Gemm {
 public:
  virtual ~Gemm() = default;
  virtual void apply(const GemmArgs<T> &args) const = 0;
};

class GemmQ4 {
 public:
  virtual ~GemmQ4() = default;
  virtual void apply(const GemmQ4Args &args) const = 0;
};

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
