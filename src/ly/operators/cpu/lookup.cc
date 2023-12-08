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

#include "lymath/lymath.h"
#include "ly/operators/cpu/copy.h"
#include "ly/operators/cpu/lookup.h"
#include "ly/operators/cpu/print.h"
#include "ly/operators/cpu/subtensor_list.h"
#include "ly/operators/cpu/tensor.h"

namespace ly {
namespace op {
namespace cpu {

Tensor lookupFp32(Subtensor<const float> table, Subtensor<const LongType> indices) {
  CHECK(table.rank() == 2 && indices.rank() == 2);

  int batch_size = indices.dimension(0);
  int seq_len = indices.dimension(1);
  int d_model = table.dimension(1);
  Tensor output = tensor(lut::makeConstSpan({batch_size, seq_len, d_model}), DType::kFloat);
  Subtensor<float> emb = Subtensor<float>::fromTensor(output);

  for (int batch = 0; batch < batch_size; ++batch) {
    Subtensor<const LongType> indices_b = indices.subtensor(batch);
    Subtensor<float> emb_b = emb.subtensor(batch);
    for (int l = 0; l < seq_len; ++l) {
      int64_t index = indices_b.elem(l);
      CHECK(index < table.dimension(0)) << "indices out of range";

      Subtensor<const float> emb_src = table.subtensor(static_cast<int>(index));
      Subtensor<float> emb_tgt = emb_b.subtensor(l);
      copyFp32(emb_src, emb_tgt);
    }
  }

  return output;
}

template<typename T>
void applyDequant(int64_t offset, int n, const internal::TensorData *data, float *tgt);

template<>
void applyDequant<QInt4Group32>(
  int64_t offset, int n, const internal::TensorData *data, float *tgt) {
  lymath_dequant_q4(
      n,
      (const lymath_q4x2_t *)data->getData<QInt4Group32>(offset),
      (const lymath_float16_t *)data->getSlot(1)->getData<Float16>(
          offset / QInt4Group32::GroupSize),
      (const uint8_t *)data->getSlot(2)->getData<UInt8>(offset / QInt4Group32::GroupSize / 2),
      tgt);
}

template<typename T>
Tensor lookupQuantized(const Tensor &embd, Subtensor<const LongType> indices) {
  CHECK(embd.getDim() == 2 && embd.getShape(1) % QInt4SymGroup32::GroupSize == 0);

  int vocabSize = embd.getShape(0);
  int embdDim = embd.getShape(1);
  const internal::TensorData *embdData = embd.getDataObject();

  std::vector<int> shapeC = indices.getShape();
  shapeC.push_back(embdDim);

  Tensor C = tensor(shapeC, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);

  SubtensorList<const LongType> vAs = getVectorList(indices);
  SubtensorList<float> vCs = getMatrixList(Cs);
  CHECK(vAs.getSize() == vCs.getSize() && vAs.getShape()[0].shape == vCs.getShape()[0].shape);
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const LongType> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      int64_t index = vA.elem(i);
      CHECK(index < vocabSize) << "indices out of range.";

      Subtensor<float> vCi = vC.subtensor(i);
      applyDequant<T>(embdDim * index, embdDim, embdData, vCi.data);
    }
  }

  return C;
}

Tensor lookup(const Tensor &table, const Tensor &indices) {
  switch (table.getDType()) {
    case DType::kFloat:
      return lookupFp32(
          Subtensor<const float>::fromTensor(table),
          Subtensor<const LongType>::fromTensor(indices));
    case DType::kQInt4Group32:
      return lookupQuantized<QInt4Group32>(table, Subtensor<const LongType>::fromTensor(indices));
    default:
      NOT_IMPL();
  }

  return Tensor();
}

}  // cpu
}  // op
}  // ly
