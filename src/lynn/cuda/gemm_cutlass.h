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

#include "lynn/cuda/gemm.h"

namespace libllm {
namespace op {
namespace cuda {

/// @brief Operators implemented by cuBLAS.
class CutlassGemm : public Gemm {
 public:
  static std::shared_ptr<Gemm> create();

  lut::ErrorCode gemmMxfp4Bf16(
      int m,
      int n,
      int k,
      float alpha,
      const Fp4E2M0x2 *A,
      const UInt8 *sfA,
      const Fp4E2M0x2 *B,
      const UInt8 *sfB,
      Float16 *C) override;
};

}  // namespace cuda
}  // namespace op
}  // namespace libllm
