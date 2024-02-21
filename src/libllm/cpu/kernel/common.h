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
#include "kernel.h"

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

typedef uint16_t *PFp16;
typedef float *PFp32;

typedef const float *PCFp32;
typedef const uint16_t *PCFp16;
typedef const int8_t *PCInt8;
typedef const UInt8 *PCUInt8;

typedef const uint8_t *PCQ4x2;

constexpr int GEMVMinRowsPerThread = 128;
constexpr int CvtMinElemPerThread = 1024;
constexpr int DequantMinElemPerThread = 1024;
constexpr int GroupSizeQ4 = 128;

class DataQ4 {
 public:
  constexpr DataQ4(): 
      _data(nullptr),
      _scale(nullptr),
      _zero(nullptr) {}
  constexpr DataQ4(PCQ4x2 data, PCFp16 scale, PCUInt8 zero) :
      _data(data),
      _scale(scale),
      _zero(zero) {}

  constexpr PCQ4x2 getDataByGroup(int64_t groupIdx) const {
    return _data + groupIdx * GroupSizeQ4 / 2;
  }
  constexpr PCFp16 getScaleByGroup(int64_t groupIdx) const { return _scale + groupIdx; }
  constexpr PCQ4x2 getZeroByGroup(int64_t groupIdx) const { return _zero + groupIdx / 2; }

  constexpr Fp16 getScaleValByGroup(int64_t groupIdx) const { return _scale[groupIdx]; }
  constexpr UInt8 getZeroValByGroup(int64_t groupIdx) const {
    return (_zero[groupIdx >> 1] >> ((groupIdx & 1) << 2)) & 0xf;
  }

 private:
  PCQ4x2 _data;
  PCFp16 _scale;
  PCUInt8 _zero;
};

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
