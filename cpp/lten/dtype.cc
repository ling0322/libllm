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

#include "lten/dtype.h"

#include "lutil/log.h"
#include "lutil/strings.h"

namespace lten {

constexpr int16_t DType::kUnknown;
constexpr int16_t DType::kFloat;
constexpr int16_t DType::kLong;
constexpr int16_t DType::kUInt8;
constexpr int16_t DType::kFloat16;
constexpr int16_t DType::kQInt4x32;
constexpr int16_t DType::kInt8;

DType::DType(int16_t dtype)
    : _dtype(dtype) {
}

template<>
DType DType::getTypeImpl<float>() {
  return DType::kFloat;
}
template<>
DType DType::getTypeImpl<int64_t>() {
  return DType::kLong;
}
template<>
DType DType::getTypeImpl<UInt8>() {
  return DType::kUInt8;
}
template<>
DType DType::getTypeImpl<Float16>() {
  return DType::kFloat16;
}
template<>
DType DType::getTypeImpl<QInt4x32>() {
  return DType::kQInt4x32;
}
template<>
DType DType::getTypeImpl<Int8>() {
  return DType::kInt8;
}

int64_t DType::getTotalSize(int64_t numel) const {
  switch (_dtype) {
    case DType::kFloat:
      return 4 * numel;
    case DType::kFloat16:
      return 2 * numel;
    case DType::kLong:
      return 8 * numel;
    case DType::kQInt4x32:
      CHECK(numel % 32 == 0);
      return numel / 32 * sizeof(QInt4x32);
    case DType::kInt8:
    case DType::kUInt8:
      return numel;
    default:
      NOT_IMPL();
      return -1;
  }
}

bool DType::isValid() const {
  switch (_dtype) {
    case DType::kFloat:
    case DType::kFloat16:
    case DType::kLong:
    case DType::kUInt8:
    case DType::kQInt4x32:
    case DType::kInt8:
      return true;
    default:
      return false;
  }
}

bool DType::isQuantized() const {
  switch (_dtype) {
    case DType::kQInt4x32:
      return true;
    default:
      return false;
  }
}

bool DType::isFloat() const {
  switch (_dtype) {
    case DType::kFloat16:
    case DType::kFloat:
      return true;
    default:
      return false;
  }
}

int DType::getGroupSize() const {
  switch (_dtype) {
    case DType::kQInt4x32:
      return 32;
    default:
      NOT_IMPL();
  }

  return 0;
}

std::string DType::toString() const {
  switch (_dtype) {
    case DType::kFloat:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kLong:
      return "int64";
    case DType::kUInt8:
      return "uint8";
    case DType::kQInt4x32:
      return "q4";
    case DType::kInt8:
      return "int8";
    default:
      return lut::sprintf("unknown(%d)", int(_dtype));
  }
}

}  // namespace lten
