cmake_minimum_required(VERSION 3.22)
project(flash_attn LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Built as a separate project so its pinned CUTLASS never reaches the libllm build.
set(FLASH_ATTN_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass")

if(NOT EXISTS "${FLASH_ATTN_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
        "FlashAttention CUTLASS was not found at ${FLASH_ATTN_CUTLASS_DIR}. "
        "Run third_party/install_flash_attn.sh.")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Real code for every target: sm80 PTX JITs into silently wrong kernels on sm120.
    set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89-real 90-real 120-real)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB FLASH_ATTN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/flash_fwd*.cu")

add_library(flash_attn STATIC
    ${FLASH_ATTN_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/flash.cc")

target_compile_features(flash_attn PRIVATE cuda_std_17)
target_compile_definitions(flash_attn PRIVATE FLASHATTENTION_DISABLE_DROPOUT)
target_compile_options(flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__
        -U__CUDA_NO_BFLOAT16_CONVERSIONS__>)
target_include_directories(flash_attn PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${FLASH_ATTN_CUTLASS_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(flash_attn PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
