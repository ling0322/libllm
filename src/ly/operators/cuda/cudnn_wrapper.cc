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

#include <cudnn.h>
#include "ly/operators/cuda/cuda_tensor_data.h"
#include "ly/operators/cuda/common.h"
#include "ly/operators/cuda/cudnn_wrapper.h"
#include "ly/operators/cpu/view.h"
#include "lyutil/log.h"

#define CHECK_CUDNN_STATUS(x) { \
      cudnnStatus_t status = x; \
      if (status != CUDNN_STATUS_SUCCESS) { \
        LOG(ERROR) << "Error while calling: " << #x; \
        throw lut::AbortedError(cudnnGetErrorString(status)); \
      } \
    }

namespace ly {
namespace op {
namespace cuda {

CudnnWrapper::CudnnWrapper() : _handle{nullptr, checkDestroy<cudnnHandle_t>(cudnnDestroy)} {}


void loggingCallback(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg) {
  if (sev == CUDNN_SEV_FATAL) LOG(FATAL) << msg;
  else if (sev == CUDNN_SEV_ERROR) LOG(ERROR) << msg;
  else if (sev == CUDNN_SEV_WARNING) LOG(WARN) << msg;
  else if (sev == CUDNN_SEV_INFO) LOG(INFO) << msg;
  else NOT_IMPL();
}

std::shared_ptr<CudnnWrapper> CudnnWrapper::create() {
  std::shared_ptr<CudnnWrapper> ops{new CudnnWrapper()};
  CHECK_CUDNN_STATUS(cudnnSetCallback(CUDNN_SEV_WARNING_EN | CUDNN_SEV_ERROR_EN,
                                      nullptr,
                                      loggingCallback));
  CHECK_CUDNN_STATUS(cudnnCreate(ops->_handle.get_pp()));

  
  return ops;
}

cudnnDataType_t CudnnWrapper::getCudnnDataType(const Tensor &tensor) {
  // Only single-slot tensor is allowed.
  const internal::TensorData *tensorData = tensor.getDataObject();
  CHECK(tensorData->getNumSlot() == 1);

  switch (tensorData->getDType()) {
    case DType::kFloat16:
      return CUDNN_DATA_HALF;
    default:
      NOT_IMPL();
  }
}

template<typename T>
std::function<void(T)> CudnnWrapper::checkDestroy(std::function<cudnnStatus_t(T)> destroyFunc) {
  return [destroyFunc](T handle) {
    cudnnStatus_t status = destroyFunc(handle);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "Error while calling destroy function: " << cudnnGetErrorString(status);
    }
  };
}

auto_handle<cudnnTensorDescriptor_t> CudnnWrapper::createCudnnTensorDescriptor(
    const Tensor &tensor) {
  auto_handle<cudnnTensorDescriptor_t> tensorDesc{
      nullptr, checkDestroy<cudnnTensorDescriptor_t>(cudnnDestroyTensorDescriptor)};
  CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(tensorDesc.get_pp()));

  if (tensor.isContiguous()) {
    int n = 1, h = 1, w = 1, c = 1;
    switch (tensor.getDim()) {
      case 1:
        c = tensor.getShape(0);
        break;
      case 2:
        w = tensor.getShape(0);
        c = tensor.getShape(1);
        break;
      case 3:
        h = tensor.getShape(0);
        w = tensor.getShape(1);
        c = tensor.getShape(2);
        break;
      case 4:
        n = tensor.getShape(0);
        h = tensor.getShape(1);
        w = tensor.getShape(2);
        c = tensor.getShape(3);
        break;
      default:
        NOT_IMPL();
    }
    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        tensorDesc.get(),
        CUDNN_TENSOR_NHWC,
        getCudnnDataType(tensor),
        n,
        c,
        h,
        w));
  } else {
    int n = 1, h = 1, w = 1, c = 1;
    int ns = 0, hs = 0, ws = 0, cs = 0;
    switch (tensor.getDim()) {
      case 1:
        c = tensor.getShape(0);
        cs = tensor.getStride(0);
        break;
      case 2:
        w = tensor.getShape(0);
        c = tensor.getShape(1);
        ws = tensor.getStride(0);
        cs = tensor.getStride(1);
        break;
      case 3:
        h = tensor.getShape(0);
        w = tensor.getShape(1);
        c = tensor.getShape(2);
        hs = tensor.getStride(0);
        ws = tensor.getStride(1);
        cs = tensor.getStride(2);
        break;
      case 4:
        n = tensor.getShape(0);
        h = tensor.getShape(1);
        w = tensor.getShape(2);
        c = tensor.getShape(3);
        ns = tensor.getStride(0);
        hs = tensor.getStride(1);
        ws = tensor.getStride(2);
        cs = tensor.getStride(3);
        break;
      default:
        NOT_IMPL();
    }
    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptorEx(
        tensorDesc.get(),
        getCudnnDataType(tensor),
        n,
        c,
        h,
        w,
        ns,
        cs,
        hs,
        ws));
  }

  return tensorDesc;
}

void CudnnWrapper::copy(Tensor src, Tensor dest) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(dest.getDevice().getType() == Device::kCuda);
  CHECK(src.getDim() <= 4);
  CHECK(src.getDType() == dest.getDType());

  float alphaFloat = 1.0;
  float betaFloat = 0.0;

  void *alpha = &alphaFloat;
  void *beta = &betaFloat;
  auto_handle<cudnnTensorDescriptor_t> srcDesc = createCudnnTensorDescriptor(src);
  auto_handle<cudnnTensorDescriptor_t> destDesc = createCudnnTensorDescriptor(dest);
  CHECK_CUDNN_STATUS(cudnnTransformTensor(
      _handle.get(),
      alpha,
      srcDesc.get(),
      src.getData<void>(),
      beta,
      destDesc.get(),
      dest.getData<void>()));
}

Tensor CudnnWrapper::scale(Tensor src, float alpha) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(src.getDim() <= 4);

  Tensor dest = createCudaTensorHalf(src.getShape());
  copy(src, dest);

  auto_handle<cudnnTensorDescriptor_t> destDesc = createCudnnTensorDescriptor(dest);
  CHECK_CUDNN_STATUS(cudnnScaleTensor(
      _handle.get(),
      destDesc.get(),
      dest.getData<void>(),
      &alpha));

  return dest;
}

Tensor CudnnWrapper::applyOp(const Tensor &A, const Tensor &B, cudnnOpTensorOp_t op) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(B.getDevice().getType() == Device::kCuda);
  CHECK(A.getDim() <= 4);
  CHECK(B.getDim() <= 4);

  auto_handle<cudnnOpTensorDescriptor_t> opDesc{
      nullptr,
      checkDestroy<cudnnOpTensorDescriptor_t>(cudnnDestroyOpTensorDescriptor)};
  CHECK_CUDNN_STATUS(cudnnCreateOpTensorDescriptor(opDesc.get_pp()));
  CHECK_CUDNN_STATUS(cudnnSetOpTensorDescriptor(
      opDesc.get(),
      op,
      CUDNN_DATA_FLOAT,
      CUDNN_PROPAGATE_NAN));
  
  float alpha = 1.0f;
  float beta = 0.0f;

  Tensor C = createCudaTensorHalf(A.getShape());
  auto_handle<cudnnTensorDescriptor_t> descA = createCudnnTensorDescriptor(A);
  auto_handle<cudnnTensorDescriptor_t> descB = createCudnnTensorDescriptor(B);
  auto_handle<cudnnTensorDescriptor_t> descC = createCudnnTensorDescriptor(C);

  CHECK_CUDNN_STATUS(cudnnOpTensor(
      _handle.get(),
      opDesc.get(),
      &alpha,
      descA.get(),
      A.getData<void>(),
      &alpha,
      descB.get(),
      B.getData<void>(),
      &beta,
      descC.get(),
      C.getData<void>()));

  return C;
}

Tensor CudnnWrapper::softmax(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  CHECK(tensor.getDim() <= 4);

  float alpha = 1.0f;
  float beta = 0.0f;

  Tensor C = createCudaTensorHalf(tensor.getShape());
  auto_handle<cudnnTensorDescriptor_t> descA = createCudnnTensorDescriptor(tensor);
  auto_handle<cudnnTensorDescriptor_t> descC = createCudnnTensorDescriptor(C);

  CHECK_CUDNN_STATUS(cudnnSoftmaxForward(
      _handle.get(),
      CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL,
      &alpha,
      descA.get(),
      tensor.getData<void>(),
      &beta,
      descC.get(),
      C.getData<void>()));
  
  return C;
}

Tensor CudnnWrapper::reduce(const Tensor &tensor, cudnnReduceTensorOp_t op) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  CHECK(tensor.getDim() <= 4);

  float alpha = 1.0f;
  float beta = 0.0f;

  auto_handle<cudnnReduceTensorDescriptor_t> reduceDesc{
      nullptr,
      checkDestroy<cudnnReduceTensorDescriptor_t>(cudnnDestroyReduceTensorDescriptor)};
  
  CHECK_CUDNN_STATUS(cudnnCreateReduceTensorDescriptor(reduceDesc.get_pp()));
  CHECK_CUDNN_STATUS(cudnnSetReduceTensorDescriptor(
      reduceDesc.get(),
      op,
      CUDNN_DATA_FLOAT,
      CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_NO_INDICES,
      CUDNN_32BIT_INDICES));

  std::vector<Tensor::ShapeType> shapeC = tensor.getShape();
  shapeC.back() = 1;
  Tensor C = createCudaTensorHalf(shapeC);
  auto_handle<cudnnTensorDescriptor_t> descA = createCudnnTensorDescriptor(tensor);
  auto_handle<cudnnTensorDescriptor_t> descC = createCudnnTensorDescriptor(C);

  size_t workSpaceSize = 0;
  CHECK_CUDNN_STATUS(cudnnGetReductionWorkspaceSize(
      _handle.get(),
      reduceDesc.get(),
      descA.get(),
      descC.get(),
      &workSpaceSize));

  lut::c_ptr<Byte> workspace;
  if (workSpaceSize) workspace = llynCudaAlloc<Byte>(workSpaceSize);

  CHECK_CUDNN_STATUS(cudnnReduceTensor(
      _handle.get(),
      reduceDesc.get(),
      nullptr,
      0,
      workspace.get(),
      workSpaceSize,
      &alpha,
      descA.get(),
      tensor.getData<void>(),
      &beta,
      descC.get(),
      C.getData<void>()));

  if (shapeC.size() > 1) 
    shapeC.pop_back();
  else
    shapeC.back() = 1;

  C = op::cpu::view(C, shapeC);
  return C;
}

}  // cuda
}  // op
}  // ly

