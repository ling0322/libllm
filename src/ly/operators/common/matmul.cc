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

#include "ly/operators/common/matmul.h"

namespace ly {
namespace op {
namespace common {

std::vector<int> getBmmOutputShape(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() >= B.getDim());
  CHECK(A.getDim() > 2 && A.getDim() <= 4 && B.getDim() >= 2);
  std::vector<int> shape;

  // broadcast B
  int broadcastDims = A.getDim() - B.getDim();
  for (int i = 0; i < broadcastDims; ++i) {
    shape.push_back(A.getShape(i));
  }

  // batch dim: B.shape(i) == A.shape(broadcastDims + i)
  int batchDims = B.getDim() - 2;
  for (int i = 0; i < batchDims; ++i) {
    CHECK(A.getShape(broadcastDims + i) == B.getShape(i));
    shape.push_back(B.getShape(i));
  }

  shape.push_back(A.getShape(-2));
  shape.push_back(B.getShape(-1));

  return shape;
}

GEMMArgs generateGemmArgs(const Tensor &A, const Tensor &B, const Tensor &C) {
  CHECK(A.getDim() >= B.getDim() && A.getDim() == C.getDim());
  CHECK(B.getDim() >= 2);
  CHECK(A.getShape(-2) == C.getShape(-2));
  CHECK(A.getShape(-1) == B.getShape(-2));
  CHECK(B.getShape(-1) == C.getShape(-1));

  bool transA, transB;
  int lda, ldb;
  if (A.getStride(-1) == 1) {
    transA = false;
    lda = A.getStride(-2);
  } else if (A.getStride(-2) == 1) {
    transA = true;
    lda = A.getStride(-1);
  } else {
    NOT_IMPL();
  }

  if (B.getStride(-1) == 1) {
    transB = false;
    ldb = B.getStride(-2);
  } else if (B.getStride(-2) == 1) {
    transB = true;
    ldb = B.getStride(-1);
  } else {
    NOT_IMPL();
  }

  int m = A.getShape(-2);
  int k = A.getShape(-1);
  int n = B.getShape(-1);
  int ldc = C.getStride(-2);

  GEMMArgs gemmArgs;
  gemmArgs.K = k;
  gemmArgs.lda = lda;
  gemmArgs.ldb = ldb;
  gemmArgs.ldc = ldc;
  gemmArgs.M = m;
  gemmArgs.N = n;
  gemmArgs.transA = transA;
  gemmArgs.transB = transB;

  return gemmArgs;
}

}  // commoon
}  // op
}  // ly
