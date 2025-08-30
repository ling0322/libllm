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

#include "libllm/cpu/print.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <inttypes.h>

#include "libllm/cpu/accessor.h"
#include "libllm/cpu/tensor_printer.h"
#include "libllm/cuda/common.h"

namespace libllm {
namespace op {
namespace cuda {

const float e2m1Fp4Values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};

struct CudaPrinterImpl {
  template<typename T, int DIM>
  using accessor_type = op::cpu::TensorAccessor<T, DIM>;

  static void printValue(accessor_type<const half, 1> valAcc, int index) {
    half hvalue;
    LL_CHECK_CUDA_STATUS(cudaMemcpy(&hvalue, &valAcc[index], sizeof(half), cudaMemcpyDeviceToHost));

    float value = hvalue;
    if (std::abs(value) > 100 || std::abs(value) < 0.01) {
      printf("%.4e", value);
    } else {
      printf("%.4f", value);
    }
  }

  static void printValue(accessor_type<const float, 1> valAcc, int index) {
    float value;
    LL_CHECK_CUDA_STATUS(cudaMemcpy(&value, &valAcc[index], sizeof(float), cudaMemcpyDeviceToHost));

    if (std::abs(value) > 100 || std::abs(value) < 0.01) {
      printf("%.4e", value);
    } else {
      printf("%.4f", value);
    }
  }

  static void printValue(accessor_type<const UInt8, 1> valAcc, int index) {
    uint8_t hvalue;
    LL_CHECK_CUDA_STATUS(
        cudaMemcpy(&hvalue, &valAcc[index], sizeof(uint8_t), cudaMemcpyDeviceToHost));
    printf("%d", static_cast<int>(hvalue));
  }

  static void printValue(accessor_type<const LongType, 1> valAcc, int index) {
    LongType v;
    LL_CHECK_CUDA_STATUS(cudaMemcpy(&v, &valAcc[index], sizeof(LongType), cudaMemcpyDeviceToHost));
    printf("%ld", v);
  }

  static void printValue(accessor_type<const Fp4E2M0x2, 1> valAcc, int index) {
    Fp4E2M0x2 hvalue;
    LL_CHECK_CUDA_STATUS(
        cudaMemcpy(&hvalue, &valAcc[index], sizeof(Fp4E2M0x2), cudaMemcpyDeviceToHost));

    printf("(%+.1f, %+.1f)", e2m1Fp4Values[hvalue.v0], e2m1Fp4Values[hvalue.v1]);
  }

  static void printValue(accessor_type<const BoolType, 1> valAcc, int index) {
    BoolType v;
    LL_CHECK_CUDA_STATUS(cudaMemcpy(&v, &valAcc[index], sizeof(BoolType), cudaMemcpyDeviceToHost));

    if (v) {
      printf("true");
    } else {
      printf("false");
    }
  }
};

void print(const Tensor &tensor) {
  op::cpu::TensorPrinter<CudaPrinterImpl> printer;

  if (tensor.getDType() == DType::kFloat16)
    printer.print<half>(tensor);
  else if (tensor.getDType() == DType::kFloat)
    printer.print<float>(tensor);
  else if (tensor.getDType() == DType::kUInt8)
    printer.print<UInt8>(tensor);
  else if (tensor.getDType() == DType::kFp4E2M0x2)
    printer.print<Fp4E2M0x2>(tensor);
  else if (tensor.getDType() == DType::kLong)
    printer.print<LongType>(tensor);
  else if (tensor.getDType() == DType::kBool)
    printer.print<BoolType>(tensor);
  else
    NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
