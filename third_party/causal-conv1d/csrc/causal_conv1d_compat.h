/******************************************************************************
 * Not upstream. Stands in for the two pieces of PyTorch the vendored kernels
 * reached for, so that they build against nothing but the CUDA toolkit:
 *
 *   at::Half / at::BFloat16  ->  the CUDA toolkit's own __half / __nv_bfloat16
 *   C10_CUDA_CHECK, C10_CUDA_KERNEL_LAUNCH_CHECK  ->  the macros below
 *
 * Errors are raised as std::runtime_error; the caller in flint/cuda turns them
 * into the engine's own error type.
 ******************************************************************************/

#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#define C10_CUDA_CHECK(expr)                                                     \
  do {                                                                           \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      throw std::runtime_error(                                                  \
          std::string("causal_conv1d: ") + cudaGetErrorString(status_));         \
    }                                                                            \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
