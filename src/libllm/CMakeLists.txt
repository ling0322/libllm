find_package(OpenMP)
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
    "lut/is_debug.cc"
    "lut/path.cc"
    "lut/random.cc"
    "lut/reader.cc"
    "lut/strings.cc"
    "lut/time.cc"
    "lut/thread_pool.cc"
    "lut/zip_file.cc")

set(libllm_SOURCES
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
    "cpu/fingerprint.cc"
    "cpu/lookup.cc"
    "cpu/matmul.cc"
    "cpu/print.cc"
    "cpu/rand.cc"
    "cpu/rms_norm.cc"
    "cpu/softmax.cc"
    "cpu/swiglu.cc"
    "cpu/tensor.cc"
    "cpu/transform.cc"
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
    "model_for_generation.cc"
    "module.cc"
    "mp.cc"
    "operators.cc"
    "qwen.cc"
    "sampler.cc"
    "state_map.cc"
    "tensor.cc"
    "tokenizer.cc"
    "../../third_party/ruapu/ruapu.cc")

set(unittest_SOURCES
    "cpu/kernel/benchmark.cc"
    "cpu/kernel/interface_test.cc"
    # "cpu/kernel/unittest_kernel.cc"
    "cpu/test.cc"
    "lut/path_test.cc"
    "lut/strings_test.cc"
    "chatglm_test.cc"
    "llama_test.cc"
    "module_test.cc"
    "operator_tester.cc"
    "tensor_test.cc"
    "test_helper.cc"
    "test_main.cc")

set(llm_SOURCES
    "dialog_manager.cc"
    "llm_main.cc")


set(benchmark_SOURCES "benchmark_main.cc")

set(libllm_INCDIR ".." "../../third_party")
set(libllm_LIBADD lut ${CMAKE_DL_LIBS})

if(WITH_OPENMP)
    if(NOT OPENMP_FOUND)
        message(FATAL_ERROR "WITH_OPENMP=ON build OpenMP package not found.")
    endif()
    set(libllm_SOURCES ${libllm_SOURCES} "mp_openmp.cc")
    set(libllm_INCDIR ${libllm_INCDIR} ${OpenMP_CXX_INCLUDE_DIRS})
    set(libllm_LIBADD ${libllm_LIBADD} OpenMP::OpenMP_CXX)
else()
    set(libllm_SOURCES ${libllm_SOURCES} "mp_thread_pool.cc")
endif()

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
        "cuda/print.cc"
        "cuda/reduce.cu"
        "cuda/rms_norm.cu"
        "cuda/softmax.cu"
        "cuda/swiglu.cu"
        "cuda/to_device.cc"
        "cuda/transform.cu")

    set(unittest_SOURCES ${unittest_SOURCES} "cuda/test.cc")

    if (WITH_CUTLASS)
        set(libllm_SOURCES ${libllm_SOURCES} "cuda/gemm_cutlass.cu")
    endif(WITH_CUTLASS)

    set(llmextcublas_SOURCES
        "cuda/gemm_cublas.cc"
        "lut/internal/log.cc")
endif()

# OS specific code
if(WIN32)
    set(libllm_SOURCES
        ${libllm_SOURCES}
        "lut/path_windows.cc"
        "lut/platform_windows.cc"
        "lut/shared_library_windows.cc")
endif()
if(UNIX)
    set(libllm_SOURCES
        ${libllm_SOURCES}
        "lut/platform_linux.cc"
        "lut/shared_library_linux.cc")
endif()
if(UNIX AND APPLE)
    set(libllm_SOURCES
        ${libllm_SOURCES}
        "lut/path_darwin.cc")
endif()
if(UNIX AND NOT APPLE)
    set(libllm_SOURCES
        ${libllm_SOURCES}
        "lut/path_linux.cc")
endif()

# CPU specific code
if(LIBLLM_KERNEL_X86_64)
    set(libllm_SOURCES
        ${libllm_SOURCES}
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
    set(unittest_SOURCES 
        ${unittest_SOURCES}
        "cpu/kernel/avx2_test.cc"
        "cpu/kernel/avx512_test.cc")
endif()
if(LIBLLM_KERNEL_AARCH64)
    set(libllm_SOURCES ${libllm_SOURCES} "cpu/kernel/asimdhp.cc")
    set(unittest_SOURCES 
        ${unittest_SOURCES}
        "cpu/test_float16.cc"
        "cpu/kernel/asimdhp_test.cc")
endif()

add_library(lut STATIC ${lut_SOURCES})
set_target_properties(lut PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(lut PRIVATE ${libllm_INCDIR})

add_library(libllm_static OBJECT ${libllm_SOURCES})
target_compile_options(libllm_static PRIVATE "-DLIBLLM_EXPORTS")
set_target_properties(libllm_static PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(libllm_static PROPERTIES CUDA_VISIBILITY_PRESET hidden)
target_include_directories(libllm_static PRIVATE ${libllm_INCDIR})

add_library(libllm SHARED $<TARGET_OBJECTS:libllm_static>)
target_link_libraries(libllm ${libllm_LIBADD}  )
set_property(TARGET libllm PROPERTY OUTPUT_NAME llm)
if(UNIX AND NOT APPLE)
    target_link_options(libllm PUBLIC "-Wl,--no-undefined")
endif()

set(unittest_LIBADD
    libllm_static
    ${libllm_LIBADD}
    catch2)
if (WITH_MKL)
    set(unittest_LIBADD
        ${unittest_LIBADD}
        mkl_intel_lp64
        mkl_intel_thread
        mkl_core
        iomp5)
endif()

set(benchmark_LIBADD
    libllm_static
    ${libllm_LIBADD})

add_library(catch2 STATIC  "../../third_party/catch2/catch_amalgamated.cpp")
add_executable(unittest ${unittest_SOURCES})
target_include_directories(unittest PRIVATE ${libllm_INCDIR})
target_link_libraries(unittest ${unittest_LIBADD})

add_executable(benchmark ${benchmark_SOURCES})
target_include_directories(benchmark PRIVATE ${libllm_INCDIR})
target_link_libraries(benchmark ${benchmark_LIBADD})

add_executable(llm ${llm_SOURCES})
target_include_directories(llm PRIVATE ..)
target_link_libraries(llm libllm lut)

if (WITH_CUDA)
    add_library(llmextcublas SHARED ${llmextcublas_SOURCES})
    target_include_directories(llmextcublas PRIVATE ${libllm_INCDIR})
    target_link_libraries(llmextcublas lut CUDA::cublas)
    if(UNIX)
        target_link_options(llmextcublas PUBLIC "-Wl,--no-undefined")
    endif(UNIX)
endif()

enable_testing()
add_test(NAME unittest COMMAND $<TARGET_FILE:unittest>)

add_custom_target(llmbin
    ALL
    DEPENDS libllm
    COMMAND go build -o $<PATH:GET_PARENT_PATH,$<TARGET_FILE:unittest>>/llm_chat${CMAKE_EXECUTABLE_SUFFIX}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/go/cmd)