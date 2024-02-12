find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(lut_SOURCES
    "lut/internal/log.cc"
    "lut/internal/sprintf.cc"
    "lut/error.cc"
    "lut/flags.cc"
    "lut/half.cc"
    "lut/ini_config.cc"
    "lut/path.cc"
    "lut/random.cc"
    "lut/reader.cc"
    "lut/strings.cc"
    "lut/time.cc")

set(libllm_SOURCES
    "cpu/kernel/kernel_avx2.cc"
    "cpu/kernel/kernel_avx512.cc"
    "cpu/kernel/kernel_fallback.cc"
    "cpu/kernel/kernel.cc"
    "cpu/kernel/util.cc"
    "cpu/apply_rotary_pos_emb.cc"
    "cpu/attention.cc"
    "cpu/cast.cc"
    "cpu/cat.cc"
    "cpu/common.cc"
    "cpu/copy.cc"
    "cpu/cpu_operators.cc"
    "cpu/cpu_tensor_data.cc"
    "cpu/fingerprint.cc"
    "cpu/lookup.cc"
    "cpu/matmul.cc"
    "cpu/mul.cc"
    "cpu/print.cc"
    "cpu/rand.cc"
    "cpu/swiglu.cc"
    "cpu/tensor.cc"
    "cpu/view.cc"
    "bpe_config.cc"
    "bpe_encoder.cc"
    "bpe_model.cc"
    "chatglm.cc"
    "c_api.cc"
    "context.cc"
    "device.cc"
    "dtype.cc"
    "functional.cc"
    "generator.cc"
    "llama.cc"
    "model_factory.cc"
    "module.cc"
    "operators.cc"
    "sampler.cc"
    "state_map.cc"
    "tensor.cc"
    "tokenizer.cc")

set(unittest_SOURCES
    "cpu/kernel/test.cc"
    "cpu/test.cc"
    "lut/path_test.cc"
    "lut/strings_test.cc"
    "chatglm_test.cc"
    "module_test.cc"
    "tensor_test.cc"
    "test_helper.cc"
    "test_main.cc")

set(llm_SOURCES
    "dialog_manager.cc"
    "llm_main.cc")

set(libllm_INCDIR "..")

if (WITH_CUDA)
    set(libllm_INCDIR ${libllm_INCDIR} ${CUDAToolkit_INCLUDE_DIRS})
    set(libllm_SOURCES
        ${libllm_SOURCES}
        "cuda/apply_rotary_pos_emb.cu"
        "cuda/binary_op.cu"
        "cuda/cast.cu"
        "cuda/causal_mask.cu"
        "cuda/copy.cu"
        "cuda/common.cc"
        "cuda/cuda_operators.cc"
        "cuda/cuda_tensor_data.cc"
        "cuda/dequant.cu"
        "cuda/lookup.cu"
        "cuda/matmul.cc"
        "cuda/matvec.cu"
        "cuda/reduce.cu"
        "cuda/rms_norm.cu"
        "cuda/softmax.cu"
        "cuda/swiglu.cu"
        "cuda/to_device.cc"
        "cuda/transform.cu")

    set(unittest_SOURCES ${unittest_SOURCES} "cuda/test.cc")

    if (WITH_CUTLASS)
        set(libllm_SOURCES ${libllm_SOURCES} "cuda/gemm_cutlass.cu")
        set(unittest_SOURCES ${unittest_SOURCES} "cuda/gemm_cutlass.cu")
    endif(WITH_CUTLASS)

    set(llmextcublas_SOURCES
        "cuda/gemm_cublas.cc"
        "lut/internal/log.cc")
endif()

if(WIN32)
    set_source_files_properties(
        "cpu/kernel/kernel_avx512.cc"
        PROPERTIES COMPILE_FLAGS /arch:AVX512)
    set_source_files_properties(
        "cpu/kernel/kernel_avx2.cc"
        PROPERTIES COMPILE_FLAGS /arch:AVX2)
    set(libllm_SOURCES
        ${libllm_SOURCES}
        "lut/path_windows.cc"
        "lut/platform_windows.cc"
        "lut/shared_library_windows.cc")
endif(WIN32)

if(UNIX)
    set_source_files_properties(
        "cpu/kernel/kernel_avx512.cc"
        PROPERTIES COMPILE_FLAGS "-mavx512f")
    set_source_files_properties(
        "cpu/kernel/kernel_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
    set(libllm_SOURCES
        ${libllm_SOURCES}
        "lut/path_linux.cc"
        "lut/platform_linux.cc"
        "lut/shared_library_linux.cc")
endif(UNIX)

add_library(lut STATIC ${lut_SOURCES})
set_target_properties(lut PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(lut PRIVATE "..")

set(libllm_LIBADD
    lut
    ${CMAKE_DL_LIBS}
    OpenMP::OpenMP_CXX)

add_library(libllm_static OBJECT ${libllm_SOURCES})
target_compile_options(libllm_static PRIVATE "-DLIBLLM_EXPORTS")
set_target_properties(libllm_static PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(libllm_static PROPERTIES CUDA_VISIBILITY_PRESET hidden)
target_include_directories(libllm_static PRIVATE ${libllm_INCDIR})

add_library(libllm SHARED $<TARGET_OBJECTS:libllm_static>)
target_link_libraries(libllm ${libllm_LIBADD}  )
set_property(TARGET libllm PROPERTY OUTPUT_NAME llm)
if(UNIX)
    target_link_options(libllm PUBLIC "-Wl,--no-undefined")
endif(UNIX)

add_library(catch2 STATIC  "../../third_party/catch2/catch_amalgamated.cpp")
add_executable(unittest ${unittest_SOURCES})
target_include_directories(unittest PRIVATE .. "../../third_party/")
target_link_libraries(unittest libllm_static lut catch2)

add_executable(llm ${llm_SOURCES})
target_include_directories(llm PRIVATE ..)
target_link_libraries(llm libllm lut)

if (WITH_CUDA)
    add_library(llmextcublas SHARED ${llmextcublas_SOURCES})
    target_include_directories(llmextcublas PRIVATE .. "../../third_party/")
    target_link_libraries(llmextcublas lut CUDA::cublas)
    if(UNIX)
        target_link_options(llmextcublas PUBLIC "-Wl,--no-undefined")
    endif(UNIX)
endif()

enable_testing()
add_test(NAME unittest COMMAND $<TARGET_FILE:unittest>)
