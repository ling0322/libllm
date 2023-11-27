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
#include "llyn/operators/cuda/cuda_tensor_data.h"
#include "llyn/operators/cuda/common.h"
#include "llyn/operators/cuda/cudnn_operators.h"
#include "lyutil/log.h"

#define CHECK_CUDNN_STATUS(x) { \
      cudnnStatus_t status = x; \
      if (status != CUDNN_STATUS_SUCCESS) { \
        LOG(ERROR) << "Error while calling: " << #x; \
        throw ly::AbortedError(cudnnGetErrorString(status)); \
      } \
    }

namespace llyn {
namespace op {
namespace cuda {

CudnnOperators::CudnnOperators() : _handle{nullptr, checkDestroy<cudnnHandle_t>(cudnnDestroy)} {}


void loggingCallback(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg) {
  if (sev == CUDNN_SEV_FATAL) LOG(FATAL) << msg;
  else if (sev == CUDNN_SEV_ERROR) LOG(ERROR) << msg;
  else if (sev == CUDNN_SEV_WARNING) LOG(WARN) << msg;
  else if (sev == CUDNN_SEV_INFO) LOG(INFO) << msg;
  else NOT_IMPL();
}

std::shared_ptr<CudnnOperators> CudnnOperators::create() {
  std::shared_ptr<CudnnOperators> ops{new CudnnOperators()};
  CHECK_CUDNN_STATUS(cudnnSetCallback(CUDNN_SEV_WARNING_EN | CUDNN_SEV_ERROR_EN,
                                      nullptr,
                                      loggingCallback));
  CHECK_CUDNN_STATUS(cudnnCreate(ops->_handle.get_pp()));

  
  return ops;
}

cudnnDataType_t CudnnOperators::getCudnnDataType(const Tensor &tensor) {
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
std::function<void(T)> CudnnOperators::checkDestroy(std::function<cudnnStatus_t(T)> destroyFunc) {
  return [destroyFunc](T handle) {
    cudnnStatus_t status = destroyFunc(handle);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "Error while calling destroy function: " << cudnnGetErrorString(status);
    }
  };
}

auto_handle<cudnnTensorDescriptor_t> CudnnOperators::createCudnnTensorDescriptor(
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

void CudnnOperators::copy(Tensor src, Tensor dest) {
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

}  // cuda
}  // op
}  // llyn

