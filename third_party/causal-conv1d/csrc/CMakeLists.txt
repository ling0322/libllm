cmake_minimum_required(VERSION 3.22)
project(causal_conv1d LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Built as a separate project, the way FlashAttention is, so its compile options stay off the
# libllm build. Unlike FlashAttention it needs nothing but the CUDA toolkit: the kernels use cub,
# which ships with it, so there is no dependency to fetch first.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89-real 90-real 120-real)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(causal_conv1d STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/causal_conv1d_fwd.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/causal_conv1d_update.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/causal_conv1d.cc")

target_compile_features(causal_conv1d PRIVATE cuda_std_17)
target_compile_options(causal_conv1d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__>)
target_include_directories(causal_conv1d PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(causal_conv1d PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
