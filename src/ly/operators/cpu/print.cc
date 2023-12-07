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

#include "ly/operators/cpu/print.h"

namespace ly {
namespace op {
namespace cpu {

constexpr int kPrintEdgeItems = 3;

template<typename T>
void printValue(T value);

template<>
void printValue(float value) {
  if (std::abs(value) > 100 || std::abs(value) < 0.01) {
    printf("%.4e", value);
  } else {
    printf("%.4f", value);
  }
}

template<>
void printValue(LongType value) {
  printf("%lld", value);
}

template<typename T>
void print1D(Subtensor<const T> A) {
  CHECK(A.rank() == 1);

  printf("[");
  for (int i = 0; i < A.dimension(0); ++i) {
    T elem = A.elem(i);
    printValue<T>(elem);

    if (A.dimension(0) > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      printf(" ... ");
      i += A.dimension(0) - kPrintEdgeItems * 2;
    } else if (i < A.dimension(0) - 1) {
      printf(", ");
    }
  }
  printf("]");
}

template<typename T>
void printND(Subtensor<const T> A, int pad_space) {
  CHECK(A.rank() >= 2);

  printf("[");
  for (int i = 0; i < A.dimension(0); ++i) {
    if (i > 0) {
      for (int j = 0; j < pad_space + 1; ++j) printf(" ");
    }
    if (A.rank() == 2) {
      print1D<T>(A.subtensor(i));
    } else {
      printND<T>(A.subtensor(i), pad_space + 1);
    }
    
    
    if (i < A.dimension(0) - 1) printf(",\n"); 
    if (A.dimension(0) > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      for (int j = 0; j < pad_space + 1; ++j) printf(" ");
      printf("...\n");
      i += A.dimension(0) - kPrintEdgeItems * 2;
    }
  }
  printf("]");
}

template<typename T>
void print(Subtensor<const T> tensor) {
  int rank = tensor.rank();

  printf("tensor(");
  switch (rank) {
    case 1:
      print1D<T>(tensor);
      break;
    default:
      printND<T>(tensor, 7);
      break;
  }
  printf(", shape=(");
  for (int d = 0; d < tensor.rank(); ++d) {
    if (d) printf(", ");
    printf("%d", tensor.dimension(d));
  }
  puts("))");
}

void print(const Tensor &tensor) {
  switch (tensor.getDType()) {
    case DType::kFloat:
      print<float>(Subtensor<const float>::fromTensor(tensor));
      break;
    case DType::kLong:
      print<LongType>(Subtensor<const LongType>::fromTensor(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Print";
  }
}

}  // cpu
}  // op
}  // ly
