add_subdirectory("lutil")

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(flint_SOURCES
    "capi.cc"
    "cpu/kernel/fallback.cc"
    "cpu/kernel/interface.cc"
    "cpu/kernel/util.cc"
    "cpu/all_close.cc"
    "cpu/binary_op.cc"
    "cpu/cast.cc"
    "cpu/common.cc"
    "cpu/copy.cc"
    "cpu/cpu_operators.cc"
    "cpu/cpu_tensor_data.cc"
    "cpu/fill.cc"
    "cpu/lookup.cc"
    "cpu/matmul.cc"
    "cpu/normalizations.cc"
    "cpu/print.cc"
    "cpu/rand.cc"
    "cpu/reduce.cc"
    "cpu/repetition_penalty.cc"
    "cpu/softmax.cc"
    "cpu/swiglu.cc"
    "cpu/tensor.cc"
    "cpu/transform.cc"
    "cpu/view.cc"
    "device.cc"
    "dtype.cc"
    "functional.cc"
    "memory.cc"
    "operators.cc"
    "tensor.cc"
    "../third_party/ruapu/ruapu.cc")

set(flint_test_SOURCES
    "capi_test.cc"
    "cpu/fill_test.cc"
    "cpu/kernel/benchmark.cc"
    "cpu/kernel/interface_test.cc"
    "cpu/lookup_test.cc"
    "cpu/matmul_test.cc"
    "cpu/reduce_test.cc"
    "cpu/sampling_test.cc"
    "cpu/softmax_test.cc"
    "tensor_test.cc")

set(flint_benchmark_SOURCES
    "cpu/benchmark.cc")

set(flint_INCDIR "." ".." "../third_party")

if(WITH_OPENMP)
    if(NOT OPENMP_FOUND)
        message(FATAL_ERROR "WITH_OPENMP=ON build OpenMP package not found.")
    endif()
    set(flint_INCDIR ${flint_INCDIR} ${OpenMP_CXX_INCLUDE_DIRS})
endif()

if (WITH_CUDA)
    set(flint_INCDIR ${flint_INCDIR} ${CUDAToolkit_INCLUDE_DIRS})
    set(flint_SOURCES
        ${flint_SOURCES}
        "cuda/arange.cu"
        "cuda/binary.cu"
        "cuda/binary_scalar.cu"
        "cuda/cast.cu"
        "cuda/causal_mask.cu"
        "cuda/copy.cu"
        "cuda/common.cc"
        "cuda/cuda_operators.cc"
        "cuda/cuda_tensor_data.cc"
        "cuda/dequant.cu"
        "cuda/fill.cu"
        "cuda/gemm_cublas.cc"
        "cuda/lookup.cu"
        "cuda/matmul.cc"
        "cuda/matvec.cu"
        "cuda/print.cc"
        "cuda/rand.cc"
        "cuda/reduce.cu"
        "cuda/repetition_penalty.cu"
        "cuda/rms_norm.cu"
        "cuda/rotary_embedding.cu"
        "cuda/sampling.cu"
        "cuda/softmax.cu"
        "cuda/store_kv_cache.cu"
        "cuda/swiglu.cu"
        "cuda/to_device.cc"
        "cuda/unary.cu")

    if(WITH_FLASH_ATTN)
        set(flint_SOURCES ${flint_SOURCES} "cuda/flash_attn.cc")
        set(flint_test_SOURCES ${flint_test_SOURCES} "cuda/flash_attn_test.cc")
    endif()

    set(flint_test_SOURCES
        ${flint_test_SOURCES}
        "cuda/arange_test.cc"
        "cuda/attention_test.cc"
        "cuda/binary_scalar_test.cc"
        "cuda/binary_test.cc"
        "cuda/causal_mask_test.cc"
        "cuda/common_test.cc"
        "cuda/copy_test.cc"
        "cuda/fill_test.cc"
        "cuda/lookup_test.cc"
        "cuda/matmul_test.cc"
        "cuda/matvec_test.cc"
        "cuda/rand_test.cc"
        "cuda/reduce_test.cc"
        "cuda/repetition_penalty_test.cc"
        "cuda/rms_norm_test.cc"
        "cuda/rotary_embedding_test.cc"
        "cuda/sampling_test.cc"
        "cuda/softmax_test.cc"
        "cuda/store_kv_cache_test.cc"
        "cuda/swiglu_test.cc"
        "cuda/to_device_test.cc")
    set(flint_benchmark_SOURCES ${flint_benchmark_SOURCES} "cuda/benchmark.cc")

    if (WITH_CUTLASS)
        set(flint_SOURCES ${flint_SOURCES}
            "cuda/gemm_cutlass.cu")
        set(flint_test_SOURCES ${flint_test_SOURCES} "cuda/gemm_cutlass_test.cc")
        set(flint_INCDIR ${flint_INCDIR}
            "../third_party/cutlass/include"
            "../third_party/cutlass/tools/util/include")
    endif(WITH_CUTLASS)
endif()

# CPU specific code
if(LIBLLM_KERNEL_X86_64)
    set(flint_SOURCES
        ${flint_SOURCES}
        "cpu/kernel/avx2.cc"
        "cpu/kernel/avx512.cc")
    if(WIN32)
        set_source_files_properties(
            "cpu/kernel/avx512.cc"
            PROPERTIES COMPILE_FLAGS /arch:AVX512)
        set_source_files_properties(
            "cpu/kernel/avx2.cc"
            PROPERTIES COMPILE_FLAGS /arch:AVX2)
    endif(WIN32)
    if(UNIX)
        set_source_files_properties(
            "cpu/kernel/avx512.cc"
            PROPERTIES COMPILE_FLAGS "-mavx512f")
        set_source_files_properties(
            "cpu/kernel/avx2.cc"
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
    endif(UNIX)
    set(flint_test_SOURCES 
        ${flint_test_SOURCES}
        "cpu/kernel/avx2_test.cc"
        "cpu/kernel/avx512_test.cc")
endif()

if(LIBLLM_KERNEL_AARCH64)
    set(flint_SOURCES ${flint_SOURCES} "cpu/kernel/asimdhp.cc")
    set(flint_test_SOURCES 
        ${flint_test_SOURCES}
        "cpu/test_float16.cc"
        "cpu/kernel/asimdhp_test.cc")
endif()

add_library(flint_static OBJECT ${flint_SOURCES})
target_compile_options(flint_static PRIVATE "-DLIBLLM_EXPORTS")
target_include_directories(flint_static PRIVATE ${LIBDWARF_INCLUDE_DIRS})
if(WITH_FLASH_ATTN)
    target_link_libraries(flint_static PRIVATE flash_attn)
endif()
#set_target_properties(flint_static PROPERTIES CXX_VISIBILITY_PRESET hidden)
#set_target_properties(flint_static PROPERTIES CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
target_include_directories(flint_static PRIVATE ${flint_INCDIR})

add_library(flint_test OBJECT ${flint_test_SOURCES})
target_include_directories(flint_test PRIVATE ${flint_INCDIR})

add_library(flint_benchmark OBJECT ${flint_benchmark_SOURCES})
target_include_directories(flint_benchmark PRIVATE ${flint_INCDIR})
