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

#include "llyn/cpu/matmul.h"

#include "lyutil/strings.h"
#include "llyn/cpu/internal.h"
#include "llyn/cpu/subtensor.h"
#include "llyn/cpu/subtensor_list.h"
#include "llyn/cpu/tensor.h"
#include "lymath/lymath.h"

#ifndef _OPENMP
#error OpenMP required
#endif

namespace llyn {
namespace cpu {

using internal::TensorData;

template<typename T>
std::vector<T> repeat(ly::Span<const T> v, int n) {
  std::vector<T> rep;
  for (int i = 0; i < n; ++i) {
    for (const T &elem : v) {
      rep.emplace_back(elem);
    }
  }

  return rep;
}

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

  Tensor C = Internal::tensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  zerosFp32(Cs);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
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
  std::vector<int> shapeC = getBmmOutputShape(A, B);

  Tensor tensorC = Internal::tensor(shapeC, DType::kFloat);
  Subtensor<float> C = Subtensor<float>::fromTensor(tensorC);
  zerosFp32(C);

  SubtensorList<const float> mAs = getMatrixList(Subtensor<const float>::fromTensor(A));
  SubtensorList<const float> mBs = getMatrixList(Subtensor<const float>::fromTensor(B));
  SubtensorList<float> mCs = getMatrixList(C);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, tensorC);

  // broadcast B.
  CHECK(mAs.getSize() == mCs.getSize());
  CHECK(mAs.getSize() % mBs.getSize() == 0);
  int nb = mAs.getSize() / mBs.getSize();
  std::vector<const float*> batchB = repeat(ly::makeConstSpan(mBs.getDataPtrList()), nb);

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


// -- q4sym ----------

Tensor gemmFp32Q4SymFp32(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2 && B.getDType() == DType::kQInt4SymGroup32);

  Tensor C = Internal::tensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  zerosFp32(Cs);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  const internal::TensorData *dataObjectB = B.getDataObject();
  lymath_qgemm_nqn_q4sym_omp(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<float>(),
      gemmArgs.lda,
      reinterpret_cast<const lymath_q4x2_t *>(dataObjectB->getData<0, QInt4SymGroup32>()),
      reinterpret_cast<const lymath_float16_t *>(dataObjectB->getData<1, Float16>()),
      Cs.data,
      gemmArgs.ldc);

  return C;
}

Tensor bmmNx2Fp32Q4SymFp32(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemmFp32Q4SymFp32(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

Tensor matmulFp32Q4SymFp32(const Tensor &A, const Tensor &B) {
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmFp32Q4SymFp32(A, B);
  } else if (A.getDim() > 2 && A.isContiguous() && B.getDim() == 2) {
    return bmmNx2Fp32Q4SymFp32(A, B);
  } else {
    NOT_IMPL();
  }

  return Tensor();
}

// -- q4 ----------

Tensor gemmFp32Q4Fp32(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2 && B.getDType() == DType::kQInt4Group32);

  Tensor C = Internal::tensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  zerosFp32(Cs);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  const internal::TensorData *dataObjectB = B.getDataObject();
  lymath_q4gemm(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<float>(),
      gemmArgs.lda,
      reinterpret_cast<const lymath_q4x2_t *>(dataObjectB->getData<0, QInt4Group32>()),
      reinterpret_cast<const lymath_float16_t *>(dataObjectB->getData<1, Float16>()),
      reinterpret_cast<const int8_t *>(dataObjectB->getData<2, Int8>()),
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
}  // flint
