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

#pragma once

#include <cudnn.h>
#include <type_traits>
#include "lyutil/c_ptr.h"
#include "llyn/tensor.h"

namespace llyn {
namespace op {
namespace cuda {

/// @brief Automatically call destroy method on destruction for cudnn handles.
/// @tparam T 
template<typename T>
using auto_handle = ly::c_ptr<typename std::remove_pointer<T>::type>;

/// @brief Operators implemented by cuDNN
class CudnnOperators {
 public:
  static std::shared_ptr<CudnnOperators> create();

  Tensor contigious(Tensor tensor);
  void copy(Tensor src, Tensor dest);

 private:
  auto_handle<cudnnHandle_t> _handle;

  CudnnOperators();
  auto_handle<cudnnTensorDescriptor_t> createCudnnTensorDescriptor(const Tensor &tensor);

  /// @brief Wrap a cudnn destroy function to perform status check.
  /// @tparam T type of handle to destory.
  /// @param destroyFunc the cudnn destroy function to wrap.
  /// @return the wrapped destroy function.
  template<typename T>
  std::function<void(T)> checkDestroy(std::function<cudnnStatus_t(T)> destroyFunc);

  /// @brief convert llyn::DType to cudnnDataType_t.
  cudnnDataType_t getCudnnDataType(const Tensor &tensor);
};

}  // cuda
}  // op
}  // llyn

