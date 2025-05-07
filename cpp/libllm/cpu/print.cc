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
#include "libllm/cpu/tensor_printer.h"

namespace libllm {
namespace op {
namespace cpu {


struct CpuPrinterImpl {
  template<typename T, int DIM>
  using accessor_type = TensorAccessor<T, DIM>;

  static void printValue(const float *pval) {
    float value = *pval;
    if (std::abs(value) > 100 || std::abs(value) < 0.01) {
      printf("%.4e", value);
    } else {
      printf("%.4f", value);
    }
  }
  
  static void printValue(const LongType *pval) {
    LongType value = *pval;
    printf("%" PRId64, value);
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  static void printValue(const Float16 *pval) {
    float value = *pval;
    printValue(&value);
  }
#endif
};

void print(const Tensor &tensor) {
  TensorPrinter<CpuPrinterImpl> printer;

  if (tensor.getDType() == DType::kFloat) printer.print<float>(tensor);
#if LUT_CPU_ARCH == LUT_AARCH64
  else if (tensor.getDType() == DType::kFloat16) printer.print<Float16>(tensor);
#endif
  else if (tensor.getDType() == DType::kLong) printer.print<LongType>(tensor);
  else NOT_IMPL();
}

}  // cpu
}  // op
}  // libllm
