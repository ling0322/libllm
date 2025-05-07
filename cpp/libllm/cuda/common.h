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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <type_traits>

#include "libllm/cuda/subtensor.h"
#include "libllm/dtype.h"
#include "libllm/tensor.h"
#include "lutil/c_ptr.h"
#include "lutil/error.h"
#include "lutil/strings.h"

#define LL_CHECK_CONTIGUOUS(x)                                           \
  {                                                                      \
    if (!x.isContiguous()) {                                             \
      LOG(FATAL) << "contiguous is required for CUDA operators: " << #x; \
    }                                                                    \
  }

#define LL_CHECK_CUDA_STATUS(x)                                                          \
  {                                                                                      \
    cudaError_t status = x;                                                              \
    if (status != cudaSuccess) {                                                         \
      LOG(ERROR) << "Error while calling: " << #x << ": " << cudaGetErrorString(status); \
      throw lut::AbortedError(cudaGetErrorString(status));                               \
    }                                                                                    \
  }

namespace libllm {
namespace op {
namespace cuda {

/// @brief A q4 quantized constant matrix (2D tensor).
struct PackedSubtensor2DQInt4x32 {
  int _numRow;
  int _numCol;

  const QInt4x32 *_data;

  __forceinline__ __device__ int getNumRow() const {
    return _numRow;
  }
  __forceinline__ __device__ int getNumCol() const {
    return _numCol;
  }
  __forceinline__ __device__ const uint8_t *getData(int groupIdx) const {
    return _data[groupIdx].data;
  }
  __forceinline__ __device__ half getScaleValue(int groupIdx) const {
    return *reinterpret_cast<const half *>(&(_data[groupIdx].scale));
  }
  __forceinline__ __device__ half getZeroValue(int groupIdx) const {
    return *reinterpret_cast<const half *>(&(_data[groupIdx].zero));
  }

  PackedSubtensor2DQInt4x32(const Tensor &tensor);
};

/// @brief Automatically call destroy method on destruction for handles.
/// @tparam T
template<typename T>
using auto_handle = lut::c_ptr<typename std::remove_pointer<T>::type>;

inline void llynCudaFree(void *ptr) {
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    LOG(ERROR) << "Error while calling cudaFree(): " << cudaGetErrorString(err);
  }
}

template<typename T>
lut::c_ptr<T> llynCudaAlloc(int64_t n) {
  T *p = nullptr;
  LL_CHECK_CUDA_STATUS(cudaMalloc(&p, n * sizeof(T)));
  return {p, llynCudaFree};
}

Tensor createCudaTensorHalf(lut::Span<const int> shape);
Tensor createCudaTensorLong(lut::Span<const int> shape);
Tensor createCudaTensorFloat(lut::Span<const int> shape);

/// @brief Split a index into dim3 object according to the shape info in `size`.
/// @param index the index to split.
/// @param size the shape info. it should have at least 3 elements. size[0] is the shape and stride
//              info for axis `z`, size[1] for `y` and size[2] for `x`.
/// @return the dim3 object.
__device__ inline dim3 splitIndexToDim3(unsigned int index, const Size *size) {
  dim3 d;
  d.x = index % size[2].shape;
  d.y = (index / size[2].shape) % size[1].shape;
  d.z = index / (size[1].shape * size[2].shape);

  return d;
}

/// @brief Get attribute of cuda device 0.
/// @param attr device attribute to query.
/// @return device attribute value.
int getCudaDeviceAttribute(cudaDeviceAttr attr);

/// @brief Get number of cuda devices available.
/// @return cuda device count.
int getCudaDeviceCount();

}  // namespace cuda
}  // namespace op
}  // namespace libllm
