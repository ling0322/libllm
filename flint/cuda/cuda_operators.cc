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
#include "flint/cpu/all_close.h"
#include "flint/cuda/cast.h"
#include "flint/cuda/causal_mask.h"
#include "flint/cuda/copy.h"
#include "flint/cuda/fill.h"
#include "flint/cuda/gated_delta_net.h"
#include "flint/cuda/flash_attn.h"
#include "flint/cuda/lookup.h"
#include "flint/cuda/matmul.h"
#include "flint/cuda/print.h"
#include "flint/cuda/rand.h"
#include "flint/cuda/reduce.h"
#include "flint/cuda/repetition_penalty.h"
#include "flint/cuda/rms_norm.h"
#include "flint/cuda/rotary_embedding.h"
#include "flint/cuda/sampling.h"
#include "flint/cuda/softmax.h"
#include "flint/cuda/store_kv_cache.h"
#include "flint/cuda/swiglu.h"
#include "flint/cuda/to_device.h"
#include "flint/cuda/unary.h"
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

Tensor CudaOperators::square(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::SQUARE, input);
}

Tensor CudaOperators::max(Tensor inputs) {
  return op::cuda::reduceLastDim(inputs, inputs.getDType(), MapReduceType::MAX);
}

Tensor CudaOperators::min(Tensor inputs) {
  return op::cuda::reduceLastDim(inputs, inputs.getDType(), MapReduceType::MIN);
}

Tensor CudaOperators::divTensor(Tensor input, Tensor other) {
  return op::cuda::applyBinaryOp(BinaryOp::DIV, input, other);
}

Tensor CudaOperators::neg(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::NEG, input);
}

Tensor CudaOperators::abs(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::ABS, input);
}

Tensor CudaOperators::exp(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::EXP, input);
}

Tensor CudaOperators::sqrt(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::SQRT, input);
}

Tensor CudaOperators::rsqrt(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::RSQRT, input);
}

Tensor CudaOperators::sigmoid(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::SIGMOID, input);
}

Tensor CudaOperators::tanh(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::TANH, input);
}

Tensor CudaOperators::relu(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::RELU, input);
}

Tensor CudaOperators::gelu(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::GELU, input);
}

Tensor CudaOperators::silu(Tensor input) {
  return op::cuda::applyUnaryOp(op::cuda::UnaryOp::SILU, input);
}

Tensor CudaOperators::subFloat(Tensor input, float other) {
  return op::cuda::applyBinaryScalarOp(BinaryScalarOp::SUB, input, other);
}

bool CudaOperators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  // The comparison itself is a host-side reduction over both tensors, so bring them over and let
  // the CPU backend do it rather than growing a kernel that would only be used by tests. The cast
  // to float is what the CPU comparison expects; half tensors go through it on the device, where
  // the copy is cheaper.
  auto toHostFloat = [](const Tensor &x) {
    return op::cuda::toCpu(x.getDType() == DType::kFloat ? x : op::cuda::cast(x, DType::kFloat));
  };

  return op::cpu::allClose(toHostFloat(A), toHostFloat(B), rtol, atol);
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

void CudaOperators::rotaryEmbedding(
    Tensor positions,
    Tensor query,
    Tensor key,
    Tensor rotaryCache) {
  cuda::rotaryEmbedding(positions, query, key, rotaryCache);
}

Tensor CudaOperators::matmul(Tensor a, Tensor b) {
  return _matmul->apply(a, b);
}

Tensor CudaOperators::matmulNarrowPrecision(Tensor A, Tensor sfA, Tensor B, Tensor sfB) {
  return _matmul->applyNarrowPrecision(A, sfA, B, sfB);
}

Tensor CudaOperators::gatedDeltaNetPrefill(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor g,
    Tensor beta,
    Tensor cuSeqlens,
    Tensor stateSlots,
    Tensor state) {
  return op::cuda::gatedDeltaNetPrefill(q, k, v, g, beta, cuSeqlens, stateSlots, state);
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

Tensor CudaOperators::causalMask(int max_len) {
  return op::cuda::causalMask(max_len);
}

Tensor CudaOperators::attention(Tensor q, Tensor k, Tensor v, bool causal) {
#ifdef LIBLLM_FLASH_ATTN_ENABLED
  Tensor output = op::cuda::flashAttention(q, k, v, causal);
  if (!output.empty()) return output;
#endif

  return Operators::attention(q, k, v, causal);
}

void CudaOperators::storeKVCache(
    Tensor k,
    Tensor v,
    Tensor keyCache,
    Tensor valueCache,
    Tensor slotMapping) {
  op::cuda::storeKVCache(k, v, keyCache, valueCache, slotMapping);
}

Tensor CudaOperators::pagedAttention(
    Tensor q,
    Tensor keyCache,
    Tensor valueCache,
    Tensor blockTable,
    Tensor cuSeqlensQ,
    Tensor seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal) {
#ifdef LIBLLM_FLASH_ATTN_ENABLED
  Tensor output = op::cuda::pagedFlashAttention(
      q,
      keyCache,
      valueCache,
      blockTable,
      cuSeqlensQ,
      seqlensK,
      maxQLen,
      maxKLen,
      causal);
  if (!output.empty()) return output;
#endif

  NOT_IMPL();
}

Tensor CudaOperators::tensor(lut::Span<const int> shape, DType dtype) {
  if (dtype == DType::kFloat16) return createCudaTensorHalf(shape);
  if (dtype == DType::kFloat) return createCudaTensorFloat(shape);
  if (dtype == DType::kUInt8) return createCudaTensorUInt8(shape);

  NOT_IMPL();
}

Tensor CudaOperators::tensorLike(Tensor input) {
  return op::cuda::tensorLike(input);
}

MemorySnapshot CudaOperators::captureMemorySnapshot() {
  size_t freeMemory = 0;
  size_t totalMemory = 0;
  LL_CHECK_CUDA_STATUS(cudaMemGetInfo(&freeMemory, &totalMemory));

  uint64_t allocatedMemory = 0;
  uint64_t peakAllocatedMemory = 0;
#ifdef LIBLLM_CUDA_MALLOC_ASYNC_ENABLED
  cudaMemPool_t memoryPool;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  LL_CHECK_CUDA_STATUS(
      cudaMemPoolGetAttribute(memoryPool, cudaMemPoolAttrUsedMemCurrent, &allocatedMemory));
  LL_CHECK_CUDA_STATUS(
      cudaMemPoolGetAttribute(memoryPool, cudaMemPoolAttrUsedMemHigh, &peakAllocatedMemory));
#endif

  return MemorySnapshot(
      static_cast<int64_t>(totalMemory),
      static_cast<int64_t>(freeMemory),
      static_cast<int64_t>(allocatedMemory),
      static_cast<int64_t>(peakAllocatedMemory));
}

void CudaOperators::resetPeakMemoryStats() {
#ifdef LIBLLM_CUDA_MALLOC_ASYNC_ENABLED
  // the high watermark of a memory pool can only be reset to zero.
  cudaMemPool_t memoryPool;
  uint64_t zero = 0;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  LL_CHECK_CUDA_STATUS(cudaMemPoolSetAttribute(memoryPool, cudaMemPoolAttrUsedMemHigh, &zero));
#endif
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

Tensor CudaOperators::sample(
    Tensor logits,
    Tensor temperatures,
    Tensor topKs,
    Tensor topPs) {
  CHECK(logits.getDim() == 2);
  int rows = logits.getShape(0);
  Tensor uniformNoise = _rand->rand({rows});
  return op::cuda::sample(logits, uniformNoise, temperatures, topKs, topPs);
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
  Tensor tensor;
  if (dtype == DType::kFloat16) {
    tensor = createCudaTensorHalf(shape);
  } else if (dtype == DType::kFloat) {
    tensor = createCudaTensorFloat(shape);
  } else {
    NOT_IMPL();
  }

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
