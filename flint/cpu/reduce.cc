// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "flint/cpu/reduce.h"

#include "flint/cpu/accessor.h"
#include "flint/cpu/tensor.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

enum class MapType { EXP_FP16_FP32, SQUARE_FP16_FP32, IDENTITY, UNKNOWN };
enum class ReduceType { SUM, MAX, MIN, UNKNOWN };

constexpr MapType getMapType(MapReduceType mapReduceType) {
  switch (mapReduceType) {
    case MapReduceType::SUM:
      return MapType::IDENTITY;
    case MapReduceType::MAX:
      return MapType::IDENTITY;
    case MapReduceType::MIN:
      return MapType::IDENTITY;
    default:
      return MapType::UNKNOWN;
  }
}

constexpr ReduceType getReduceType(MapReduceType mapReduceType) {
  switch (mapReduceType) {
    case MapReduceType::SUM:
      return ReduceType::SUM;
    case MapReduceType::MAX:
      return ReduceType::MAX;
    case MapReduceType::MIN:
      return ReduceType::MIN;
    default:
      return ReduceType::UNKNOWN;
  }
}

template<typename T, ReduceType REDUCE_TYPE>
T getReduceInitial() {
  switch (REDUCE_TYPE) {
    case ReduceType::SUM:
      return T(0);
    case ReduceType::MAX:
      return -std::numeric_limits<float>::infinity();
    case ReduceType::MIN:
      return std::numeric_limits<float>::infinity();
    default:
      NOT_IMPL();
  }
}

template<typename T, ReduceType REDUCE_TYPE>
Tensor reduceKernel(Tensor A) {
  std::vector<int> shape = A.getShape();
  shape.back() = 1;
  Tensor C = tensor(shape, A.getDType());

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  int numRows = vA.getLength();
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < numRows; ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    float accumulator = getReduceInitial<T, REDUCE_TYPE>();
    for (int i = 0; i < a.getShape(0); i++) {
      if (REDUCE_TYPE == ReduceType::SUM) {
        accumulator += a[i];
      } else if (REDUCE_TYPE == ReduceType::MAX) {
        if (a[i] > accumulator) accumulator = a[i];
      } else if (REDUCE_TYPE == ReduceType::MIN) {
        if (a[i] < accumulator) accumulator = a[i];
      } else {
        NOT_IMPL();
      }
    }

    c[0] = accumulator;
  }

  shape.pop_back();
  if (shape.empty()) return C;

  return C.view(shape);
}

Tensor reduce(const Tensor &A, MapReduceType reduceType) {
  if (A.getDType() == DType::kFloat && reduceType == MapReduceType::SUM)
    return reduceKernel<float, ReduceType::SUM>(A);
  if (A.getDType() == DType::kFloat && reduceType == MapReduceType::MAX)
    return reduceKernel<float, ReduceType::MAX>(A);
  if (A.getDType() == DType::kFloat && reduceType == MapReduceType::MIN)
    return reduceKernel<float, ReduceType::MIN>(A);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16 && reduceType == MapReduceType::SUM)
    return reduceKernel<Float16, ReduceType::SUM>(A);
  if (A.getDType() == DType::kFloat16 && reduceType == MapReduceType::MAX)
    return reduceKernel<Float16, ReduceType::MAX>(A);
  if (A.getDType() == DType::kFloat16 && reduceType == MapReduceType::MIN)
    return reduceKernel<Float16, ReduceType::MIN>(A);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
