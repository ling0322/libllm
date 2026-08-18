#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace FLASH_NAMESPACE {

inline void flash_cuda_check(cudaError_t status, const char *file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) + ": "
        + cudaGetErrorString(status));
}

} // namespace FLASH_NAMESPACE

#define FLASH_CUDA_CHECK(EXPR) \
    FLASH_NAMESPACE::flash_cuda_check((EXPR), __FILE__, __LINE__)
#define FLASH_CUDA_KERNEL_LAUNCH_CHECK() FLASH_CUDA_CHECK(cudaGetLastError())
