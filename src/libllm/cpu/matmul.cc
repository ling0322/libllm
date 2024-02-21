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

#include "libllm/cpu/matmul.h"

#include "libllm/lut/strings.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/subtensor.h"
#include "libllm/cpu/subtensor_list.h"
#include "libllm/cpu/tensor.h"
#include "libllm/cpu/kernel/kernel.h"

#ifndef _OPENMP
#error OpenMP required
#endif

namespace libllm {
namespace op {
namespace cpu {

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

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  kernel::sgemm(
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
      gemmArgs.ldc,
      kernel::Mode::OMP);

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
  if (A.getDim() != B.getDim()) xB = expandBatchDims(B, A.getShape());
  std::vector<int> shapeC = getBmmOutputShape(A, xB);

  Tensor tensorC = cpu::tensor(shapeC, DType::kFloat);
  Subtensor<float> C = Subtensor<float>::fromTensor(tensorC);
  zerosFp32(C);

  SubtensorList<const float> mAs = getMatrixList(Subtensor<const float>::fromTensor(A));
  SubtensorList<const float> mBs = getMatrixList(Subtensor<const float>::fromTensor(xB));
  SubtensorList<float> mCs = getMatrixList(C);

  GEMMArgs gemmArgs = generateGemmArgs(A, xB, tensorC);

  // broadcast B.
  CHECK(mAs.getSize() == mCs.getSize());
  CHECK(mAs.getSize() % mBs.getSize() == 0);

  const float *const *mAp = mAs.getDataPtrList().data();
  const float *const *mBp = mBs.getDataPtrList().data();
  float *const *mCp = mCs.getDataPtrList().data();

  #pragma omp parallel for
  for (int i = 0; i < mAs.getSize(); ++i) {
    kernel::sgemm(
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
        gemmArgs.ldc,
        kernel::Mode::SingleThread);
  }

  return tensorC;
}

Tensor bmmFp32QInt4Fp32(const Tensor &A, const Tensor &B) {
  NOT_IMPL();
  return Tensor();
}

// -- q4 ----------

Tensor gemmFp32Q4Fp32(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2 && B.getDType() == DType::kQ4);

  Tensor C = cpu::tensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  zerosFp32(Cs);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  const TensorData *dataObjectB = B.getDataObject();
  kernel::gemmQ4(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<float>(),
      gemmArgs.lda,
      reinterpret_cast<const kernel::Q4x2 *>(dataObjectB->getData<Q4>()),
      reinterpret_cast<const kernel::Fp16 *>(dataObjectB->getSlot(1)->getData<Float16>()),
      reinterpret_cast<const kernel::UInt8 *>(dataObjectB->getSlot(2)->getData<UInt8>()),
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

}  // cpu
}  // op
}  // ly
