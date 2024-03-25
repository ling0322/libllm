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
#include <memory>
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/kernel/gemm_kernel.h"
#include "libllm/cpu/kernel/gemv_s.h"
#include "libllm/cpu/kernel/kernel_s.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

/// @brief Provides GEMM interface with dispatcher for GEMM/GEMV.
template<class TGemmKernel, class TGemvKernel, typename T>
class GemmImpl : public Gemm<T> {
 public:
  void apply(const GemmArgs<T> &args) const override {
    if (args.M == 1) {
      applyGemvRowVectorA(args);
    } else if (args.N == 1) {
      applyGemvColumnVectorB(args);
    } else {
      TGemmKernel().apply(args);
    }
  }

 private:
  // row vector and matrix multiplication using SGEMV.
  void applyGemvRowVectorA(const GemmArgs<T> &args) const;

  // row vector and matrix multiplication using SGEMV.
  void applyGemvColumnVectorB(const GemmArgs<T> &args) const;
};

template<class TGemmKernel, class TGemvKernel, typename T>
void GemmImpl<TGemmKernel, TGemvKernel, T>::applyGemvRowVectorA(const GemmArgs<T> &args) const {
  CHECK(args.M == 1);

  // fill C with zero.
  std::fill(args.C, args.C + args.N, 0.0f);

  TGemvKernel().apply(GemvArgs<T>{
    !args.transB,
    args.transB ? args.N : args.K,
    args.transB ? args.K : args.N,
    args.B,
    args.ldb,
    args.A,
    args.transA ? args.lda : 1,
    args.C,
    1});
}

template<class TGemmKernel, class TGemvKernel, typename T>
void GemmImpl<TGemmKernel, TGemvKernel, T>::applyGemvColumnVectorB(const GemmArgs<T> &args) const {
  CHECK(args.N == 1);

  bool needPackC = args.ldc != 1;
  if (args.ldc != 1) {
    NOT_IMPL();
  } else {
    std::fill(args.C, args.C + args.M, 0.0f);
  }

  TGemvKernel().apply(GemvArgs<T>{
      args.transA,
      args.transA ? args.K : args.M,
      args.transA ? args.M : args.K,
      args.A,
      args.lda,
      args.B,
      args.transB ? 1 : args.ldb,
      args.C,
      args.ldc});
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
