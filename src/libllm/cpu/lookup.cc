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

#include "libllm/cpu/lookup.h"

#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/copy.h"
#include "libllm/cpu/print.h"
#include "libllm/cpu/accessor.h"
#include "libllm/cpu/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T>
Tensor lookupKernel2D(const Tensor &table, const Tensor &indices) {
  CHECK(table.getDim() == 2 && indices.getDim() == 2);

  int vocabSize = table.getShape(0);
  int d0 = indices.getShape(0);
  int d1 = indices.getShape(1);
  int embdDim = table.getShape(1);
  Tensor xC = tensor(lut::makeConstSpan({d0, d1, embdDim}), DType::getType<T>());

  TensorAccessor<const T, 2> A = table;
  TensorAccessor<const LongType, 2> B = indices;
  TensorAccessor<T, 3> C = xC;

  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      int64_t index = B[i][j];
      CHECK(index < vocabSize) << "indices out of range";

      copyVector(C[i][j], A[index]);
    }
  }

  return xC;
}

template<typename SrcT, typename DestT>
Tensor lookupQuantizedKernel2D(const Tensor &table, const Tensor &indices) {
  CHECK(table.getDim() == 2 && table.getShape(1) % SrcT::GroupSize == 0);
  const TensorData *embdData = table.getDataObject();

  int vocabSize = table.getShape(0);
  int d0 = indices.getShape(0);
  int d1 = indices.getShape(1);
  int embdDim = table.getShape(1);
  Tensor xC = tensor(lut::makeConstSpan({d0, d1, embdDim}), DType::getType<DestT>());

  TensorAccessor<const LongType, 2> B = indices;
  TensorAccessor<DestT, 3> C = xC;

  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      int64_t index = B[i][j];
      CHECK(index < vocabSize) << "indices out of range";

      applyDequant<SrcT>(embdDim * index, embdDim, embdData, C[i][j].getData());
    }
  }

  return xC;
}

Tensor lookup(const Tensor &table, const Tensor &indices) {
  if (table.getDType() == DType::kFloat) return lookupKernel2D<float>(table, indices);
  if (table.getDType() == DType::kQ4)
    return lookupQuantizedKernel2D<Q4, float>(table, indices);

  NOT_IMPL();
}

}  // cpu
}  // op
}  // libllm
