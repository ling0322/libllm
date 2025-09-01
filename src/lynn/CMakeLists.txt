if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(lynn_SOURCES
    "cpu/kernel/fallback.cc"
    "cpu/kernel/interface.cc"
    "cpu/kernel/util.cc"
    "cpu/all_close.cc"
    "cpu/apply_rotary_pos_emb.cc"
    "cpu/binary_op.cc"
    "cpu/cast.cc"
    "cpu/common.cc"
    "cpu/copy.cc"
    "cpu/cpu_operators.cc"
    "cpu/cpu_tensor_data.cc"
    "cpu/fill.cc"
    "cpu/fingerprint.cc"
    "cpu/gelu.cc"
    "cpu/log_mel_spectrogram.cc"
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
    "cpu/unfold.cc"
    "cpu/view.cc"
    "context.cc"
    "device.cc"
    "dtype.cc"
    "functional.cc"
    "module.cc"
    "mp.cc"
    "operators.cc"
    "state_map.cc"
    "tensor.cc"
    "tensor_shape.cc"
    "../../third_party/ruapu/ruapu.cc")

set(lynn_test_SOURCES
    "cpu/kernel/benchmark.cc"
    "cpu/kernel/interface_test.cc"
    "cpu/test.cc"
    "module_test.cc"
    "operator_tester.cc"
    "tensor_test.cc"
    "test_helper.cc")


set(lynn_INCDIR ".." "../../third_party")

if(WITH_OPENMP)
    if(NOT OPENMP_FOUND)
        message(FATAL_ERROR "WITH_OPENMP=ON build OpenMP package not found.")
    endif()
    set(lynn_SOURCES ${lynn_SOURCES} "mp_openmp.cc")
    set(lynn_INCDIR ${lynn_INCDIR} ${OpenMP_CXX_INCLUDE_DIRS})
else()
    set(lynn_SOURCES ${lynn_SOURCES} "mp_thread_pool.cc")
endif()

if (WITH_CUDA)
    set(lynn_INCDIR ${lynn_INCDIR} ${CUDAToolkit_INCLUDE_DIRS})
    set(lynn_SOURCES
        ${lynn_SOURCES}
        "cuda/arange.cu"
        "cuda/apply_rotary_pos_emb.cu"
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
        "cuda/layer_norm.cu"
        "cuda/lookup.cu"
        "cuda/matmul.cc"
        "cuda/matvec.cu"
        "cuda/print.cc"
        "cuda/rand.cc"
        "cuda/reduce.cu"
        "cuda/repetition_penalty.cu"
        "cuda/rms_norm.cu"
        "cuda/softmax.cu"
        "cuda/swiglu.cu"
        "cuda/to_device.cc"
        "cuda/unary.cu"
        "cuda/unfold.cu")

    set(lynn_test_SOURCES ${lynn_test_SOURCES} "cuda/test.cc")

    if (WITH_CUTLASS)
        set(lynn_SOURCES ${lynn_SOURCES} "cuda/gemm_cutlass_mxfp4.cu")
        set(lynn_INCDIR ${lynn_INCDIR}
            "../../third_party/cutlass/include"
            "../../third_party/cutlass/tools/util/include")
    endif(WITH_CUTLASS)

    set(llmplugincublas_SOURCES
        "cuda/gemm_cublas.cc"
        "../lutil/internal/log.cc")
endif()

# CPU specific code
if(LIBLLM_KERNEL_X86_64)
    set(lynn_SOURCES
        ${lynn_SOURCES}
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
    set(lynn_test_SOURCES 
        ${lynn_test_SOURCES}
        "cpu/kernel/avx2_test.cc"
        "cpu/kernel/avx512_test.cc")
endif()

if(LIBLLM_KERNEL_AARCH64)
    set(lynn_SOURCES ${lynn_SOURCES} "cpu/kernel/asimdhp.cc")
    set(lynn_test_SOURCES 
        ${lynn_test_SOURCES}
        "cpu/test_float16.cc"
        "cpu/kernel/asimdhp_test.cc")
endif()

if (WITH_CUDA)
    add_library(llmplugincublas_static OBJECT ${llmplugincublas_SOURCES})
    target_compile_options(llmplugincublas_static PRIVATE "-DLIBLLM_EXPORTS")
    target_include_directories(llmplugincublas_static PRIVATE ${lynn_INCDIR})
endif()

if (WITH_FFMPEG)
    add_library(llmpluginffmpeg_static OBJECT "read_audio_ffmpeg.cc")
    target_compile_options(llmpluginffmpeg_static PRIVATE "-DLIBLLM_EXPORTS")
    target_include_directories(llmpluginffmpeg_static PRIVATE  ${lynn_INCDIR} "../../third_party/ffmpeg/")
endif()

add_library(lynn_static OBJECT ${lynn_SOURCES})
target_compile_options(lynn_static PRIVATE "-DLIBLLM_EXPORTS")
target_include_directories(lynn_static PRIVATE ${LIBDWARF_INCLUDE_DIRS})
#set_target_properties(lynn_static PROPERTIES CXX_VISIBILITY_PRESET hidden)
#set_target_properties(lynn_static PROPERTIES CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
target_include_directories(lynn_static PRIVATE ${lynn_INCDIR})

add_library(lynn_test OBJECT ${lynn_test_SOURCES})
target_include_directories(lynn_test PRIVATE ${lynn_INCDIR})
