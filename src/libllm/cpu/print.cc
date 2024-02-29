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

#include "libllm/cpu/print.h"

#include <inttypes.h>
#include "libllm/cpu/accessor.h"

namespace libllm {
namespace op {
namespace cpu {

constexpr int kPrintEdgeItems = 4;

template<typename T>
void printValue(T value);

template<>
inline void printValue(float value) {
  if (std::abs(value) > 100 || std::abs(value) < 0.01) {
    printf("%.4e", value);
  } else {
    printf("%.4f", value);
  }
}

template<>
inline void printValue(LongType value) {
  printf("%" PRId64, value);
}

template<typename T, int DIM>
typename std::enable_if<(DIM > 1), void>::type printImplND(
    TensorAccessor<const T, DIM> A,
    int padSpace) {
  printf("[");
  for (int i = 0; i < A.getShape(0); ++i) {
    if (i > 0) {
      for (int j = 0; j < padSpace + 1; ++j) printf(" ");
    }
    
    printImplND<T>(A[i], padSpace + 1);

    if (i < A.getShape(0) - 1) printf(",\n"); 
    if (A.getShape(0) > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      for (int j = 0; j < padSpace + 1; ++j) printf(" ");
      printf("...\n");
      i += A.getShape(0) - kPrintEdgeItems * 2;
    }
  }
  printf("]");
}

template<typename T, int DIM>
typename std::enable_if<DIM == 1, void>::type printImplND(
    TensorAccessor<const T, DIM> A,
    int padSpace) {
  printf("[");
  for (int i = 0; i < A.getShape(0); ++i) {
    T elem = A[i];
    printValue<T>(elem);

    if (A.getShape(0) > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      printf(" ... ");
      i += A.getShape(0) - kPrintEdgeItems * 2;
    } else if (i < A.getShape(0) - 1) {
      printf(", ");
    }
  }
  printf("]");
}

template<typename T, int DIM>
void printImpl(TensorAccessor<const T, DIM> tensor) {
  printf("tensor(");
  printImplND<T>(tensor, 7);

  printf(", shape=(");
  for (int d = 0; d < DIM; ++d) {
    if (d) printf(", ");
    printf("%d", tensor.getShape(d));
  }
  puts("))");
}

template<typename T>
void printT(const Tensor &tensor) {
  if (tensor.getDim() == 1) printImpl<T, 1>(tensor);
  if (tensor.getDim() == 2) printImpl<T, 2>(tensor);
  if (tensor.getDim() == 3) printImpl<T, 3>(tensor);
  if (tensor.getDim() == 4) printImpl<T, 4>(tensor);
  if (tensor.getDim() == 5) printImpl<T, 5>(tensor);
  if (tensor.getDim() > 5) NOT_IMPL();
}

void print(const Tensor &tensor) {
  if (tensor.getDType() == DType::kFloat) printT<float>(tensor);
  else if (tensor.getDType() == DType::kLong) printT<LongType>(tensor);
  else NOT_IMPL();
}

}  // cpu
}  // op
}  // libllm
