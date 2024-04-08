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

#include "libllm/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

struct GEMMArgs {
  bool transA;
  bool transB;
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldc;
};

std::vector<int> getBmmOutputShape(const Tensor &A, const Tensor &B);

// generate GEMMArgs from the input tensor A, B and output tensor C. dimensions of A could be
// greater than 2 (for BMM). throw exception if shape mismatch.
GEMMArgs generateGemmArgs(const Tensor &A, const Tensor &B, const Tensor &C);

Tensor matmul(const Tensor &A, const Tensor &B);

// q4
Tensor matmulFp32Q4Fp32(const Tensor &A, const Tensor &B);
Tensor gemmFp32Q4Fp32(const Tensor &A, const Tensor &B);
Tensor bmmNx2Fp32Q4Fp32(const Tensor &A, const Tensor &B);


}  // cpu
}  // op
}  // ly
