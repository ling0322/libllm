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

#pragma once

#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cpu {

template<class TPrinterImpl>
class TensorPrinter {
 public:
  static constexpr int kPrintEdgeItems = 4;

  template<typename T, int DIM>
  using accessor_type = typename TPrinterImpl::template accessor_type<const T, DIM>;

  template<typename T>
  void print(const Tensor &x) {
    if (x.getDim() == 1)
      printTensorAndInfo<T, 1>(x);
    else if (x.getDim() == 2)
      printTensorAndInfo<T, 2>(x);
    else if (x.getDim() == 3)
      printTensorAndInfo<T, 3>(x);
    else if (x.getDim() == 4)
      printTensorAndInfo<T, 4>(x);
    else if (x.getDim() == 5)
      printTensorAndInfo<T, 5>(x);
    else
      NOT_IMPL();
  }

 private:
  template<typename T, int DIM>
  typename std::enable_if<(DIM > 1), void>::type printTensor(
      accessor_type<const T, DIM> A,
      int padSpace) {
    printf("[");
    for (int i = 0; i < A.getShape(0); ++i) {
      if (i > 0) {
        for (int j = 0; j < padSpace + 1; ++j) printf(" ");
      }

      printTensor<T, DIM - 1>(A[i], padSpace + 1);

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
  typename std::enable_if<DIM == 1, void>::type printTensor(
      accessor_type<const T, DIM> A,
      int padSpace) {
    printf("[");
    for (int i = 0; i < A.getShape(0); ++i) {
      TPrinterImpl::printValue(A, i);

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
  void printTensorAndInfo(accessor_type<const T, DIM> tensor) {
    printf("tensor(");
    printTensor<T, DIM>(tensor, 7);

    printf(", shape=(");
    for (int d = 0; d < DIM; ++d) {
      if (d) printf(", ");
      printf("%d", tensor.getShape(d));
    }
    printf("), dtype=%s)\n", DType::getType<T>().toString().c_str());
  }
};

}  // namespace cpu
}  // namespace op
}  // namespace ly
