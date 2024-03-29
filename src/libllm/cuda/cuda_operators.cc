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

#include "libllm/cuda/cuda_operators.h"

#include <math.h>
#include "libllm/cuda/apply_rotary_pos_emb.h"
#include "libllm/cuda/binary_op.h"
#include "libllm/cuda/cast.h"
#include "libllm/cuda/causal_mask.h"
#include "libllm/cuda/copy.h"
#include "libllm/cuda/lookup.h"
#include "libllm/cuda/matmul.h"
#include "libllm/cuda/print.h"
#include "libllm/cuda/softmax.h"
#include "libllm/cuda/swiglu.h"
#include "libllm/cuda/to_device.h"
#include "libllm/cuda/rms_norm.h"
#include "libllm/cuda/transform.h"

#include "libllm/functional.h"

namespace libllm {
namespace op {
namespace cuda {

bool CudaOperators::isAvailable() {
  return getCudaDeviceCount() > 0;
}

Operators *CudaOperators::create() {
  std::unique_ptr<CudaOperators> op{new CudaOperators()};
  op->_matmul = MatMul::create();

  LOG(INFO) << "cuda numDevices = " << getCudaDeviceCount();
  LOG(INFO) << "cuda:0 maxThreadsPerMultiProcessor = " 
            << getCudaDeviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  LOG(INFO) << "cuda:0 multiProcessorCount = " 
            << getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);

  return op.release();
}

Tensor CudaOperators::lookup(Tensor table, Tensor indices) {
  return cuda::lookup(table, indices);
}

Tensor CudaOperators::matmul(Tensor a, Tensor b) {
  return _matmul->apply(a, b);
}

Tensor CudaOperators::mul(Tensor input, float other) {
  return op::cuda::transform(input, other, 0.0f);
}

Tensor CudaOperators::mul(Tensor input, Tensor other) {
  return op::cuda::binaryOp(input, other, BinaryOp::MUL);
}

Tensor CudaOperators::softmax(Tensor input) {
  return op::cuda::softmax(input);
}

Tensor CudaOperators::add(Tensor input, Tensor other) {
  return op::cuda::binaryOp(input, other, BinaryOp::ADD);
}

Tensor CudaOperators::rmsNorm(Tensor input, Tensor weight, float eps) {
  return op::cuda::rmsNorm(input, weight, eps);
}

Tensor CudaOperators::causalMask(int max_len) {
  return op::cuda::causalMask(max_len);
}

Tensor CudaOperators::applRotaryPosEmb(Tensor A, Tensor roPE) {
  return op::cuda::applyRotaryPosEmb(A, roPE);
}

Tensor CudaOperators::tensor(lut::Span<const int> shape, DType dtype) {
  if (dtype == DType::kFloat16) return createCudaTensorHalf(shape);

  NOT_IMPL();
}

Tensor CudaOperators::tensorLike(Tensor input) {
  CHECK(input.getDevice().getType() == Device::kCuda);

  if (input.getDType() == DType::kFloat16) return createCudaTensorHalf(input.getShape());
  if (input.getDType() == DType::kLong) return createCudaTensorLong(input.getShape());

  NOT_IMPL();
}

void CudaOperators::copy(Tensor src, Tensor dest) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(dest.getDevice().getType() == Device::kCuda);
  CHECK(src.getDType() == dest.getDType());
  src.throwIfInvalidShape(dest.getShape());

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

Tensor CudaOperators::toDevice(Tensor tensor, Device device) {
  return cuda::toDevice(tensor, device);
}

Tensor CudaOperators::cast(Tensor tensor, DType dtype) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  return cuda::cast(tensor, dtype);
}

DType CudaOperators::getDefaultFloatType() {
  return DType::kFloat16;
}

}  // cuda
}  // op
}  // ly

libllm::Operators *llynCreateCudaOperators() {
  if (libllm::op::cuda::CudaOperators::isAvailable()) {
    return libllm::op::cuda::CudaOperators::create();
  } else {
    LOG(INFO) << "No CUDA device available.";
    return nullptr;
  } 
}
