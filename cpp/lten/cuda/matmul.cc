// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include "lten/cuda/matmul.h"

#include <cuda_fp16.hpp>

#include "lten/cpu/common.h"
#include "lten/cpu/matmul.h"
#include "lten/cuda/common.h"
#include "lten/cuda/dequant.h"
#include "lten/cuda/gemm_cutlass.h"
#include "lten/cuda/matvec.h"
#include "lten/dtype.h"
#include "lutil/strings.h"

namespace lten {
namespace op {
namespace cuda {

std::shared_ptr<MatMul> MatMul::create() {
  std::shared_ptr<MatMul> mm;
  std::string err0, err1;

  try {
    mm = createCublas();
    LOG(INFO) << "Use GEMM from cuBLAS.";
    return mm;
  } catch (const lut::Error &e) {
    LOG(DEBUG) << "Load cublas extension failed with message: " << e.what();
    err0 = e.what();
  }

  try {
    mm = createCutlass();
    LOG(INFO) << "Use GEMM from cutlass.";
    return mm;
  } catch (const lut::Error &e) {
    LOG(DEBUG) << "Load cublas extension failed with message: " << e.what();
    err1 = e.what();
  }

  throw lut::AbortedError("unable to create MatMul operator: " + err0 + "; " + err1);
}

std::shared_ptr<MatMul> MatMul::createCublas() {
  std::shared_ptr<MatMul> mm{new MatMul()};

  mm->_gemmExtLib = lut::SharedLibrary::open("llmplugincublas");

  std::function<op::cuda::Gemm *()> factory;
  std::function<void(op::cuda::Gemm *)> deleter;
  factory = mm->_gemmExtLib->getFunc<op::cuda::Gemm *()>("llmGemmExt_New");
  deleter = mm->_gemmExtLib->getFunc<void(op::cuda::Gemm *)>("llmGemmExt_Delete");

  mm->_gemm = std::shared_ptr<op::cuda::Gemm>(factory(), deleter);
  if (!mm->_gemm) throw lut::AbortedError("unable to create MatMul operator.");

  return mm;
}

std::shared_ptr<MatMul> MatMul::createCutlass() {
  std::shared_ptr<MatMul> mm{new MatMul()};

#ifdef LIBLLM_CUTLASS_ENABLED
  mm->_gemm = CutlassGemm::create();
#else
  throw lut::AbortedError("Cutlass MatMul operator not enabled.");
#endif

  return mm;
}

Tensor MatMul::apply(const Tensor &A, const Tensor &B) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(B.getDevice().getType() == Device::kCuda);

  if (A.getDType() == DType::kFloat16 && B.getDType() == DType::kFloat16) {
    return matmulHalf(A, B);
  } else if (A.getDType() == DType::kFloat16 && B.getDType() == DType::kQInt4x32) {
    return matmulQ4(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor MatMul::matmulQ4(const Tensor &A, const Tensor &B) {
  CHECK(B.getDType() == DType::kQInt4x32);
  CHECK(A.getDType() == DType::kFloat16);

  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmQ4(A, B);
  } else if (A.getDim() > 2 && B.getDim() == 2) {
    return bmmToGemmQ4(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor MatMul::bmmToGemmQ4(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemmQ4(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

Tensor MatMul::gemmQ4(const Tensor &A, const Tensor &B) {
  if (A.getShape(0) > 1) {
    Tensor xB = dequantQ4ToHalf(B);
    return matmulHalf(A, xB);
  } else {
    CHECK(B.getDim() == 2 && B.getStride(0) == 1);  // make sure B is transposed.

    // use GEMV if A is a single row matrix.
    return op::cuda::gemvQ4(B.transpose(0, 1), A.transpose(0, 1)).transpose(0, 1);
  }
}

Tensor MatMul::matmulHalf(const Tensor &A, const Tensor &B) {
  CHECK(A.getDType() == B.getDType() && A.getDType() == DType::kFloat16);
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmHalf(A, B);
  } else if (A.getDim() > 2 && B.getDim() == 2 && A.isContiguous()) {
    return bmmToGemmHalf(A, B);
  } else if (A.getDim() >= 2 && B.getDim() >= 2 && A.getDim() >= B.getDim()) {
    return bmmHalf(A, B);
  } else {
    NOT_IMPL();
  }
}

template<int DIM>
std::vector<const half *> getBatchImpl(const Tensor &A);

template<>
std::vector<const half *> getBatchImpl<1>(const Tensor &A) {
  const half *base = A.getData<half>();

  int stride0 = A.getStride(0);
  std::vector<const half *> batch;
  for (int i = 0; i < A.getShape(0); ++i) {
    batch.push_back(base + i * stride0);
  }
  return batch;
}

template<>
std::vector<const half *> getBatchImpl<2>(const Tensor &A) {
  const half *base = A.getData<half>();

  int stride0 = A.getStride(0);
  int stride1 = A.getStride(1);
  std::vector<const half *> batch;
  for (int i = 0; i < A.getShape(0); ++i) {
    for (int j = 0; j < A.getShape(1); ++j) {
      batch.push_back(base + i * stride0 + j * stride1);
    }
  }
  return batch;
}

std::vector<const half *> MatMul::getBatch(const Tensor &A, int nBatchDim) {
  if (nBatchDim == 1) return getBatchImpl<1>(A);
  if (nBatchDim == 2) return getBatchImpl<2>(A);

  NOT_IMPL();
}

Tensor MatMul::bmmToGemmHalf(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemmHalf(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

Tensor MatMul::bmmHalf(Tensor A, Tensor B) {
  Tensor xB = B;
  if (A.getDim() != B.getDim()) xB = op::cpu::expandBatchDims(B, A.getShape());

  std::vector<int> shapeC = op::cpu::getBmmOutputShape(A, xB);
  Tensor C = createCudaTensorHalf(shapeC);

  int nBatchDim = A.getDim() - 2;

  op::cpu::GEMMArgs gemmArgs = op::cpu::generateGemmArgs(A, xB, C);
  std::vector<const half *> batchA = getBatch(A, nBatchDim);
  std::vector<const half *> batchB = getBatch(xB, nBatchDim);
  std::vector<const half *> batchC = getBatch(C, nBatchDim);
  CHECK(batchA.size() == batchB.size() && batchA.size() == batchC.size());

  int64_t nb = batchA.size();
  lut::c_ptr<const half *> arrayA = llynCudaAlloc<const half *>(nb);
  lut::c_ptr<const half *> arrayB = llynCudaAlloc<const half *>(nb);
  lut::c_ptr<half *> arrayC = llynCudaAlloc<half *>(nb);

  int64_t nc = sizeof(void *) * nb;
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayA.get(), batchA.data(), nc, cudaMemcpyHostToDevice));
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayB.get(), batchB.data(), nc, cudaMemcpyHostToDevice));
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayC.get(), batchC.data(), nc, cudaMemcpyHostToDevice));

  if (lut::ErrorCode::OK != _gemm->hgemmArray(
                                gemmArgs.transA,
                                gemmArgs.transB,
                                gemmArgs.M,
                                gemmArgs.N,
                                gemmArgs.K,
                                1.0f,
                                arrayA.get(),
                                gemmArgs.lda,
                                arrayB.get(),
                                gemmArgs.ldb,
                                0.0f,
                                arrayC.get(),
                                gemmArgs.ldc,
                                nb)) {
    THROW(Aborted, "hgemmArray failed.");
  }

  cudaDeviceSynchronize();
  return C;
}

Tensor MatMul::gemmHalf(Tensor A, Tensor B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);
  Tensor C = createCudaTensorHalf({A.getShape(0), B.getShape(1)});

  op::cpu::GEMMArgs gemmArgs = op::cpu::generateGemmArgs(A, B, C);
  if (lut::ErrorCode::OK != _gemm->hgemm(
                                gemmArgs.transA,
                                gemmArgs.transB,
                                gemmArgs.M,
                                gemmArgs.N,
                                gemmArgs.K,
                                1.0f,
                                A.getData<half>(),
                                gemmArgs.lda,
                                B.getData<half>(),
                                gemmArgs.ldb,
                                0.0f,
                                C.getData<half>(),
                                gemmArgs.ldc)) {
    THROW(Aborted, "hgemm failed.");
  }
  cudaDeviceSynchronize();

  return C;
}

}  // namespace cuda
}  // namespace op
}  // namespace lten
