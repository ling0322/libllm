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
constexpr int GroupSizeQ4 = 128;

class DataQ4 {
 public:
  constexpr DataQ4(): 
      _data(nullptr),
      _scale(nullptr),
      _zero(nullptr) {}
  constexpr DataQ4(const UInt4x2 *data, const Float16 *scale, const UInt4x2 *zero) :
      _data(data),
      _scale(scale),
      _zero(zero) {}

  constexpr const UInt4x2 *getDataByGroup(int64_t groupIdx) const {
    return _data + groupIdx * GroupSizeQ4 / 2;
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

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
