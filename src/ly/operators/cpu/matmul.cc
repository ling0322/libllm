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

#include "ly/operators/cpu/matmul.h"

#include "lyutil/strings.h"
#include "ly/operators/common/common.h"
#include "ly/operators/common/matmul.h"
#include "ly/operators/cpu/subtensor.h"
#include "ly/operators/cpu/subtensor_list.h"
#include "ly/operators/cpu/tensor.h"
#include "lymath/lymath.h"

#ifndef _OPENMP
#error OpenMP required
#endif

namespace ly {
namespace op {
namespace cpu {

using internal::TensorData;

Tensor matmulFp32(const Tensor &A, const Tensor &B) {
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmFp32(A, B);
  } else if (A.getDim() > 2 && A.isContiguous() && B.getDim() == 2) {
    return bmmNx2Fp32(A, B);
  } else if (A.getDim() >= 2 && B.getDim() >= 2) {
    return bmmFp32(A, B);
  } else {
    NOT_IMPL();
  }

  return Tensor();
}

Tensor gemmFp32(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);

  Tensor C = cpu::tensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  zerosFp32(Cs);

  common::GEMMArgs gemmArgs = common::generateGemmArgs(A, B, C);
  lymath_sgemm_omp(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<float>(),
      gemmArgs.lda,
      B.getData<float>(),
      gemmArgs.ldb,
      Cs.data,
      gemmArgs.ldc);

  return C;
}

Tensor bmmNx2Fp32(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemmFp32(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

Tensor bmmFp32(const Tensor &A, const Tensor &B) {
  Tensor xB = B;
  if (A.getDim() != B.getDim()) xB = common::broadcastTensor(B, A);
  std::vector<int> shapeC = common::getBmmOutputShape(A, xB);

  Tensor tensorC = cpu::tensor(shapeC, DType::kFloat);
  Subtensor<float> C = Subtensor<float>::fromTensor(tensorC);
  zerosFp32(C);

  SubtensorList<const float> mAs = getMatrixList(Subtensor<const float>::fromTensor(A));
  SubtensorList<const float> mBs = getMatrixList(Subtensor<const float>::fromTensor(xB));
  SubtensorList<float> mCs = getMatrixList(C);

  common::GEMMArgs gemmArgs = common::generateGemmArgs(A, xB, tensorC);

  // broadcast B.
  CHECK(mAs.getSize() == mCs.getSize());
  CHECK(mAs.getSize() % mBs.getSize() == 0);

  const float *const *mAp = mAs.getDataPtrList().data();
  const float *const *mBp = mBs.getDataPtrList().data();
  float *const *mCp = mCs.getDataPtrList().data();

  #pragma omp parallel for
  for (int i = 0; i < mAs.getSize(); ++i) {
    lymath_sgemm(
        gemmArgs.transA,
        gemmArgs.transB,
        gemmArgs.M,
        gemmArgs.N,
        gemmArgs.K,
        mAp[i],
        gemmArgs.lda,
        mBp[i],
        gemmArgs.ldb,
        mCp[i],
        gemmArgs.ldc);
  }

  return tensorC;
}

Tensor bmmFp32QInt4Fp32(const Tensor &A, const Tensor &B) {
  NOT_IMPL();
  return Tensor();
}

// -- q4 ----------

Tensor gemmFp32Q4Fp32(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2 && B.getDType() == DType::kQInt4Group32);

  Tensor C = cpu::tensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  zerosFp32(Cs);

  common::GEMMArgs gemmArgs = common::generateGemmArgs(A, B, C);
  const internal::TensorData *dataObjectB = B.getDataObject();
  lymath_q4gemm(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<float>(),
      gemmArgs.lda,
      reinterpret_cast<const lymath_q4x2_t *>(dataObjectB->getData<QInt4Group32>()),
      reinterpret_cast<const lymath_float16_t *>(dataObjectB->getSlot(1)->getData<Float16>()),
      reinterpret_cast<const uint8_t *>(dataObjectB->getSlot(2)->getData<UInt8>()),
      Cs.data,
      gemmArgs.ldc);

  return C;
}

Tensor bmmNx2Fp32Q4Fp32(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemmFp32Q4Fp32(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

Tensor matmulFp32Q4Fp32(const Tensor &A, const Tensor &B) {
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmFp32Q4Fp32(A, B);
  } else if (A.getDim() > 2 && A.isContiguous() && B.getDim() == 2) {
    return bmmNx2Fp32Q4Fp32(A, B);
  } else {
    NOT_IMPL();
  }

  return Tensor();
}

}  // cpu
}  // op
}  // ly
