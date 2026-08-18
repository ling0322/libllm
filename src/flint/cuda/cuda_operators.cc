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

#include "flint/cuda/cuda_operators.h"

#include <math.h>

#include "flint/cuda/arange.h"
#include "flint/cuda/binary.h"
#include "flint/cuda/binary_scalar.h"
#include "flint/cuda/cast.h"
#include "flint/cuda/causal_mask.h"
#include "flint/cuda/copy.h"
#include "flint/cuda/fill.h"
#include "flint/cuda/layer_norm.h"
#include "flint/cuda/lookup.h"
#include "flint/cuda/matmul.h"
#include "flint/cuda/print.h"
#include "flint/cuda/rand.h"
#include "flint/cuda/reduce.h"
#include "flint/cuda/repetition_penalty.h"
#include "flint/cuda/rms_norm.h"
#include "flint/cuda/sampling.h"
#include "flint/cuda/softmax.h"
#include "flint/cuda/swiglu.h"
#include "flint/cuda/to_device.h"
#include "flint/cuda/unary.h"
#include "flint/cuda/unfold.h"
#include "flint/functional.h"

namespace fl {
namespace op {
namespace cuda {

bool CudaOperators::isAvailable() {
  return getCudaDeviceCount() > 0;
}

std::shared_ptr<Operators> CudaOperators::create(int options) {
  std::shared_ptr<CudaOperators> op{new CudaOperators()};
  if (!isAvailable()) {
    LOG(INFO) << "No CUDA device available.";
    return nullptr;
  }

#ifdef LIBLLM_CUDA_MALLOC_ASYNC_ENABLED
  int memoryPoolsSupported = 0;
  LL_CHECK_CUDA_STATUS(
      cudaDeviceGetAttribute(&memoryPoolsSupported, cudaDevAttrMemoryPoolsSupported, 0));
  CHECK(memoryPoolsSupported) << "CUDA device does not support stream-ordered memory allocation";

  cudaMemPool_t memoryPool;
  uint64_t releaseThreshold = UINT64_MAX;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  LL_CHECK_CUDA_STATUS(
      cudaMemPoolSetAttribute(memoryPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
#endif

  if (options & OPT_CUBLAS_GEMM) {
    LOG(INFO) << "Create CUDA operators with CUBLAS only";
    op->_matmul = MatMul::createCublas();
  } else if (options & OPT_CUTLASS_GEMM) {
    LOG(INFO) << "Create CUDA operators with CUTLASS only";
    op->_matmul = MatMul::createCutlass();
  } else {
    op->_matmul = MatMul::create();
  }
  op->_rand = Rand::newRand();

  LOG(INFO) << "cuda numDevices = " << getCudaDeviceCount();
  LOG(INFO) << "cuda:0 maxThreadsPerMultiProcessor = "
            << getCudaDeviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  LOG(INFO) << "cuda:0 multiProcessorCount = "
            << getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);

  return op;
}

void CudaOperators::fill(Tensor input, float value) {
  return op::cuda::fill(input, value);
}

Tensor CudaOperators::gelu(Tensor input) {
  return op::cuda::applyUnaryOp(UnaryOp::GELU, input);
}

Tensor CudaOperators::square(Tensor input) {
  return op::cuda::applyUnaryOp(UnaryOp::SQUARE, input);
}

Tensor CudaOperators::max(Tensor inputs) {
  return op::cuda::reduceLastDim(inputs, inputs.getDType(), MapReduceType::MAX);
}

bool CudaOperators::all(Tensor A) {
  return op::cuda::elemBool(op::cuda::reduceAll(A, DType::kBool, MapReduceType::ALL));
}

Tensor CudaOperators::sum(Tensor inputs, int dim) {
  Tensor C;

  if (dim == -1 || dim == inputs.getDim() - 1) {
    C = op::cuda::reduceLastDim(inputs, DType::kFloat, MapReduceType::SUM);
  } else if (dim == None) {
    C = op::cuda::reduceAll(inputs, DType::kFloat, MapReduceType::SUM);
  }

  if (inputs.getDType() == DType::kFloat16) {
    C = castFloatToHalf(C);
  }
  return C;
}

Tensor CudaOperators::lookup(Tensor table, Tensor indices) {
  return cuda::lookup(table, indices);
}

Tensor CudaOperators::matmul(Tensor a, Tensor b) {
  return _matmul->apply(a, b);
}

Tensor CudaOperators::matmulNarrowPrecision(Tensor A, Tensor sfA, Tensor B, Tensor sfB) {
  return _matmul->applyNarrowPrecision(A, sfA, B, sfB);
}

Tensor CudaOperators::mul(Tensor input, float other) {
  return op::cuda::applyBinaryScalarOp(BinaryScalarOp::MUL, input, other);
}

Tensor CudaOperators::div(Tensor input, float other) {
  return op::cuda::applyBinaryScalarOp(BinaryScalarOp::DIV, input, other);
}

Tensor CudaOperators::mod(Tensor input, LongType other) {
  return op::cuda::applyBinaryScalarOpLong(BinaryScalarOp::MOD, input, other);
}

Tensor CudaOperators::mul(Tensor input, Tensor other) {
  return op::cuda::applyBinaryOp(BinaryOp::MUL, input, other);
}

Tensor CudaOperators::softmax(Tensor input) {
  return op::cuda::softmax(input);
}

Tensor CudaOperators::add(Tensor input, Tensor other) {
  return op::cuda::applyBinaryOp(BinaryOp::ADD, input, other);
}

Tensor CudaOperators::sub(Tensor input, Tensor other) {
  return op::cuda::applyBinaryOp(BinaryOp::SUB, input, other);
}

void CudaOperators::repetitionPenalty(Tensor logits, Tensor history, float weight) {
  CHECK(history.getDType() == DType::kLong);

  return op::cuda::repetitionPenalty(logits, history, weight);
}

Tensor CudaOperators::rmsNorm(Tensor input, Tensor weight, float eps) {
  return op::cuda::rmsNorm(input, weight, eps);
}

Tensor CudaOperators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  return op::cuda::layerNorm(input, weight, bias, eps);
}

Tensor CudaOperators::causalMask(int max_len) {
  return op::cuda::causalMask(max_len);
}

Tensor CudaOperators::tensor(lut::Span<const int> shape, DType dtype) {
  if (dtype == DType::kFloat16) return createCudaTensorHalf(shape);
  if (dtype == DType::kUInt8) return createCudaTensorUInt8(shape);

  NOT_IMPL();
}

Tensor CudaOperators::unfold(Tensor input, int kernelSize, int stride) {
  return op::cuda::unfold(input, kernelSize, stride);
}

Tensor CudaOperators::tensorLike(Tensor input) {
  return op::cuda::tensorLike(input);
}

void CudaOperators::copy(Tensor src, Tensor dest) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(dest.getDevice().getType() == Device::kCuda);
  CHECK(src.getDType() == dest.getDType());
  src.throwIfInvalidShape(dest.getShape(), "CudaOperators::copy");

  if (src.isContiguous() && dest.isContiguous()) {
    copyContig(src, dest);
  } else {
    op::cuda::copy(src, dest);
  }
}

void CudaOperators::print(Tensor tensor) {
  op::cuda::print(tensor);
}

Tensor CudaOperators::swiglu(Tensor A) {
  return op::cuda::swiglu(A);
}

Tensor CudaOperators::sample(Tensor distribution, int topK, float topP) {
  int vocabSize = distribution.getShape(-1);
  int64_t rows64 = distribution.getNumEl() / vocabSize;
  CHECK(rows64 <= std::numeric_limits<int>::max());
  int rows = static_cast<int>(rows64);
  Tensor uniformNoise = _rand->rand({rows, topK});
  return op::cuda::sample(distribution, uniformNoise, topK, topP);
}

Tensor CudaOperators::to(Device device, Tensor tensor) {
  return cuda::toDevice(device, tensor);
}

Tensor CudaOperators::cast(Tensor tensor, DType dtype) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  return cuda::cast(tensor, dtype);
}

DType CudaOperators::getDefaultFloatType() {
  return DType::kFloat16;
}

Tensor CudaOperators::zeros(lut::Span<const int> shape, DType dtype) {
  Tensor tensor = createCudaTensorHalf(shape);
  op::cuda::fill(tensor, 0.0);

  return tensor;
}

Tensor CudaOperators::randNormal(lut::Span<const int> shape) {
  return _rand->randNormal(shape);
}

Tensor CudaOperators::rand(lut::Span<const int> shape, DType dtype) {
  Tensor r = _rand->rand(shape);
  return cuda::cast(r, dtype);
}

Tensor CudaOperators::arangeLong(LongType begin, LongType end, LongType step) {
  return cuda::arangeLong(begin, end, step);
}

float CudaOperators::elem(Tensor tensor) {
  return op::cuda::elem(tensor);
}

bool CudaOperators::elemBool(Tensor tensor) {
  return op::cuda::elemBool(tensor);
}

Tensor CudaOperators::eq(Tensor input, Tensor other) {
  return op::cuda::applyBinaryOp(BinaryOp::EQUAL, input, other);
}

void CudaOperators::manualSeed(uint64_t seed) {
  _rand->setSeed(seed);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
