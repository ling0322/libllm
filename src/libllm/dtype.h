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

#include <string>
#include <type_traits>

#include "libllm/lut/attributes.h"

namespace libllm {

#if defined(LUT_ARCH_AARCH64)
typedef _Float16 Float16;
#elif defined(LUT_ARCH_AMD64)
struct Float16 {
  uint16_t v;
};
#else
#error unknown CPU architecture
#endif
static_assert(sizeof(Float16) == 2, "invalid size of Float16");

struct Int8 {
  int8_t v;
};
static_assert(sizeof(Int8) == 1, "invalid size of Int8");

struct UInt8 {
  uint8_t v;
};
static_assert(sizeof(Int8) == 1, "invalid size of UInt8");

struct QInt4x32 {
  static constexpr int GroupSize = 32;

  Float16 zero;
  Float16 scale;
  uint8_t data[16];
};
static_assert(sizeof(QInt4x32) == 20, "invalid size of QInt4x32");

typedef int8_t Byte;
typedef int64_t LongType;
typedef int64_t LongType;

class DType {
 public:
  static constexpr int16_t kUnknown = 0;
  static constexpr int16_t kFloat = 1;
  static constexpr int16_t kLong = 2;
  static constexpr int16_t kUInt8 = 3;
  static constexpr int16_t kFloat16 = 4;
  static constexpr int16_t kQInt4x32 = 5;
  static constexpr int16_t kInt8 = 6;

  // get DType from type T
  template<typename T>
  static DType getType();

  DType(int16_t dtype);

  bool operator==(DType rhs) const {
    return _dtype == rhs._dtype;
  }
  bool operator==(int16_t rhs) const {
    return _dtype == rhs;
  }

  operator int16_t() const {
    return _dtype;
  }

  // get the total size of specific number of elements with dtype.
  int64_t getTotalSize(int64_t numel) const;

  // dtype name in string.
  std::string toString() const;

  // return true if the dtype is valid.
  bool isValid() const;

  // return true if the dtype represents a quanztized data.
  bool isQuantized() const;

  /// @brief Return true if DType is a pure float type (without quantization).
  /// @return if DType is a pure float type.
  bool isFloat() const;

  // get the group size for quantized data.
  int getGroupSize() const;

 private:
  int16_t _dtype;

  template<typename T>
  static DType getTypeImpl();
};

template<typename T>
inline DType DType::getType() {
  return DType::getTypeImpl<typename std::remove_cv<T>::type>();
}

}  // namespace libllm
