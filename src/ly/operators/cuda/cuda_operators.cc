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

#include "ly/operators/cuda/cuda_operators.h"

#include "ly/operators/cuda/apply_rotary_pos_emb.h"
#include "ly/operators/cuda/binary_op.h"
#include "ly/operators/cuda/cast.h"
#include "ly/operators/cuda/causal_mask.h"
#include "ly/operators/cuda/copy.h"
#include "ly/operators/cuda/cudnn_wrapper.h"
#include "ly/operators/cuda/lookup.h"
#include "ly/operators/cuda/matmul.h"
#include "ly/operators/cuda/softmax.h"
#include "ly/operators/cuda/swiglu.h"
#include "ly/operators/cuda/to_device.h"
#include "ly/operators/cuda/rms_norm.h"
#include "ly/operators/cuda/transform.h"

namespace ly {
namespace op {
namespace cuda {

bool CudaOperators::isAvailable() {
  return getCudaDeviceCount() > 0;
}

internal::Operators *CudaOperators::create() {
  std::unique_ptr<CudaOperators> op{new CudaOperators()};
  op->_cudnn = CudnnWrapper::create();
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

Tensor CudaOperators::cat(Tensor A, Tensor B, int dim) {
  CHECK(A.getDType() == B.getDType() && A.getDType() == DType::kFloat16);
  dim = A.getShape_()->getRealDim(dim);
  CHECK(A.getDim() == B.getDim() &&  dim < A.getDim());

  std::vector<Tensor::ShapeType> shape = A.getShape();
  Tensor::ShapeType dA = A.getShape(dim);
  Tensor::ShapeType dB = B.getShape(dim);
  shape[dim] = dA + dB;

  Tensor C = createCudaTensorHalf(shape);
  Tensor sA = C.slice(dim, {0, dA});
  Tensor sB = C.slice(dim, {dA, dA + dB});

  copy(A, sA);
  copy(B, sB);

  return C;
}

Tensor CudaOperators::applRotaryPosEmb(Tensor A, Tensor roPE) {
  return op::cuda::applyRotaryPosEmb(A, roPE);
}

Tensor CudaOperators::createTensorLike(Tensor input) {
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

Tensor CudaOperators::attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
  Tensor scores = matmul(q, k.transpose(-2, -1));
  scores = mul(scores,  1.0f / sqrtf(1.0f * q.getShape(-1)));

  if (!mask.empty()) scores = add(scores, mask);
  scores = softmax(scores);

  Tensor outputs = matmul(scores, v);  
  return outputs;
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

ly::internal::Operators *llynCreateCudaOperators() {
  if (ly::op::cuda::CudaOperators::isAvailable()) {
    return ly::op::cuda::CudaOperators::create();
  } else {
    LOG(INFO) << "No CUDA device available.";
    return nullptr;
  } 
}
