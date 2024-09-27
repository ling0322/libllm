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

#include "libllm/cpu/accessor.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/kernel/interface.h"
#include "libllm/cpu/tensor.h"
#include "libllm/mp.h"
#include "lutil/strings.h"

namespace libllm {
namespace op {
namespace cpu {

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

template<typename T>
void callGemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    T *C,
    int ldc,
    kernel::Mode mode);

template<>
inline void callGemm<float>(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc,
    kernel::Mode mode) {
  return kernel::gemmFloat(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, mode);
}

template<>
inline void callGemm<Float16>(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const Float16 *B,
    int ldb,
    Float16 *C,
    int ldc,
    kernel::Mode mode) {
  const kernel::Float16 *xA = reinterpret_cast<const kernel::Float16 *>(A);
  const kernel::Float16 *xB = reinterpret_cast<const kernel::Float16 *>(B);
  kernel::Float16 *xC = reinterpret_cast<kernel::Float16 *>(C);
  return kernel::gemmHalf(transA, transB, M, N, K, xA, lda, xB, ldb, xC, ldc, mode);
}

template<typename T>
Tensor gemm(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);

  Tensor C = op::cpu::zeros({A.getShape(0), B.getShape(1)}, DType::getType<T>());

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  callGemm<T>(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<T>(),
      gemmArgs.lda,
      B.getData<T>(),
      gemmArgs.ldb,
      C.getData<T>(),
      gemmArgs.ldc,
      kernel::Mode::OMP);

  return C;
}

template<typename T>
Tensor bmmNx2(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemm<T>(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

template<typename T>
Tensor bmm(const Tensor &A, const Tensor &B) {
  Tensor xB = B;
  if (A.getDim() != B.getDim()) xB = expandBatchDims(B, A.getShape());
  std::vector<int> shapeC = getBmmOutputShape(A, xB);

  Tensor C = op::cpu::zeros(shapeC, DType::getType<T>());

  TensorList<const T, 2> mA = TensorList<const T, 2>::fromTensor(A);
  TensorList<const T, 2> mB = TensorList<const T, 2>::fromTensor(xB);
  TensorList<T, 2> mC = TensorList<T, 2>::fromTensor(C);

  GEMMArgs gemmArgs = generateGemmArgs(A, xB, C);

  // broadcast B.
  CHECK(mA.getLength() == mC.getLength());
  CHECK(mA.getLength() % mB.getLength() == 0);

  const T *const *mAp = mA.getDataPtrList().data();
  const T *const *mBp = mB.getDataPtrList().data();
  T *const *mCp = mC.getDataPtrList().data();

  MP::parallelFor({mA.getLength()}, [mAp, mBp, mCp, gemmArgs](MP::Partition partition) {
    for (int i : partition.getRange()) {
      callGemm<T>(
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
  });

  return C;
}

template<typename T>
Tensor matmulFloat(const Tensor &A, const Tensor &B) {
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemm<T>(A, B);
  } else if (A.getDim() > 2 && A.isContiguous() && B.getDim() == 2) {
    return bmmNx2<T>(A, B);
  } else if (A.getDim() >= 2 && B.getDim() >= 2) {
    return bmm<T>(A, B);
  } else {
    NOT_IMPL();
  }

  return Tensor();
}

template<typename T>
void callGemmQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const T *A,
    int lda,
    const kernel::QInt4x32 *B,
    T *C,
    int ldc,
    kernel::Mode mode);

template<>
inline void callGemmQInt4<float>(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const kernel::QInt4x32 *B,
    float *C,
    int ldc,
    kernel::Mode mode) {
  return kernel::gemmFloatQInt4(transA, transB, M, N, K, A, lda, B, C, ldc, mode);
}

template<>
inline void callGemmQInt4<Float16>(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const kernel::QInt4x32 *B,
    Float16 *C,
    int ldc,
    kernel::Mode mode) {
  const kernel::Float16 *xA = reinterpret_cast<const kernel::Float16 *>(A);
  kernel::Float16 *xC = reinterpret_cast<kernel::Float16 *>(C);
  return kernel::gemmHalfQInt4(transA, transB, M, N, K, xA, lda, B, xC, ldc, mode);
}

template<typename T>
Tensor gemmQInt4(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2 && B.getDType() == DType::kQInt4x32);

  Tensor C = op::cpu::zeros({A.getShape(0), B.getShape(1)}, DType::getType<T>());

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  const TensorData *dataObjectB = B.getDataObject();
  callGemmQInt4(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<T>(),
      gemmArgs.lda,
      reinterpret_cast<const kernel::QInt4x32 *>(dataObjectB->getData<QInt4x32>()),
      C.getData<T>(),
      gemmArgs.ldc,
      kernel::Mode::OMP);

  return C;
}

template<typename T>
Tensor bmmNx2QInt4(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemmQInt4<T>(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

template<typename T>
Tensor matmulQInt4(const Tensor &A, const Tensor &B) {
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmQInt4<T>(A, B);
  } else if (A.getDim() > 2 && A.isContiguous() && B.getDim() == 2) {
    return bmmNx2QInt4<T>(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor matmul(const Tensor &A, const Tensor &B) {
  DType typeA = A.getDType();
  DType typeB = B.getDType();

  if (typeA == DType::kFloat && typeB == DType::kFloat) return matmulFloat<float>(A, B);
  if (typeA == DType::kFloat && typeB == DType::kQInt4x32) return matmulQInt4<float>(A, B);

#if LUT_CPU_ARCH == LUT_AARCH64
  if (typeA == DType::kFloat16 && typeB == DType::kFloat16) return matmulFloat<Float16>(A, B);
  if (typeA == DType::kFloat16 && typeB == DType::kQInt4x32) return matmulQInt4<Float16>(A, B);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
