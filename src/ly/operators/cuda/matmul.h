// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "ly/tensor.h"
#include "ly/operators/cuda/common.h"
#include "ly/operators/cuda/gemm.h"
#include "lyutil/shared_library.h"

namespace ly {
namespace op {
namespace cuda {

class MatMul {
 public:
  static std::shared_ptr<MatMul> create();

  Tensor apply(const Tensor &A, const Tensor &B);

 protected:
  std::shared_ptr<Gemm> _gemm;
  std::shared_ptr<lut::SharedLibrary> _gemmExtLib;

  Tensor gemmHalf(Tensor A, Tensor B);
  Tensor bmmHalf(Tensor A, Tensor B);

  Tensor matmulQ4(const Tensor &A, const Tensor &B);
  Tensor gemmQ4(const Tensor &A, const Tensor &B);
  Tensor bmmToGemmQ4(const Tensor &A, const Tensor &B);

  Tensor matmulHalf(const Tensor &A, const Tensor &B);
  Tensor bmmToGemmHalf(const Tensor &A, const Tensor &B);

  std::vector<const half *> getBatch(const Tensor &A, int nBatchDim);
};

}  // cuda
}  // op
}  // ly
