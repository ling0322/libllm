cmake_minimum_required(VERSION 3.22)
project(libllm LANGUAGES CXX C)
include(CheckTypeSize)

option(WITH_CUDA "CUDA compute support." OFF)
option(WITH_MKL "MKL support (only for reference)." OFF)
option(WITH_OPENMP "Build with OpenMP." ON)
option(WITH_CUTLASS "build MatMul operators with CUTLASS." OFF)
option(WITH_FFMPEG "build the ffmpeg read audio plugin." OFF)
option(MKL_PREFIX "Prefix for MKL headers and libraries." "/opt/intel/mkl")

find_package(OpenMP)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(WITH_CUDA)
    add_definitions("-DLIBLLM_CUDA_ENABLED")
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS}) 
    set(CMAKE_CUDA_ARCHITECTURES 61)
    enable_language(CUDA)
    if(WIN32)
        set(cuda_LIBRARIES CUDA::cudart_static)
    endif(WIN32)
    if(UNIX)
        set(cuda_LIBRARIES CUDA::cudart_static)
    endif(UNIX)
endif()

if(WITH_CUTLASS)
    add_definitions("-DLIBLLM_CUTLASS_ENABLED")
    include_directories(${CUTLASS_ROOT}/include)

    set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
    if (NOT EXISTS ${CUTLASS_ROOT}/include/cutlass/blas3.h)
        message(FATAL_ERROR 
          "invalid CUTLASS_ROOT \"${CUTLASS_ROOT}\": unable to find file "
          "${CUTLASS_ROOT}/include/cutlass/blas3.h, please "
          "call install_cutlass.sh in third_party first.")
    endif()
endif()

if(WITH_MKL)
    add_definitions("-DMKL_ENABLED")
    include_directories(${MKL_PREFIX}/include)
    link_directories(${MKL_PREFIX}/lib/intel64)
    if (NOT EXISTS ${MKL_PREFIX}/include/mkl.h)
        message(FATAL_ERROR 
        "invalid MKL_PREFIX \"${MKL_PREFIX}\": unable to find file "
        "${MKL_PREFIX}/include/mkl.h, please "
        "enusre MKL_PREFIX contains the MKL \"include\" directory.")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

if(WIN32)
    add_definitions( "/D_CRT_SECURE_NO_WARNINGS /DCATCH_AMALGAMATED_CUSTOM_MAIN /DCATCH_CONFIG_PREFIX_ALL" )
endif(WIN32)
if(UNIX)
    add_definitions( "-DCATCH_AMALGAMATED_CUSTOM_MAIN -DCATCH_CONFIG_PREFIX_ALL -D_FILE_OFFSET_BITS=64" )
    set(CMAKE_CXX_FLAGS "-O3 -g")
    set(CMAKE_C_FLAGS "-O3 -g")
endif(UNIX)

message("CMAKE_HOST_SYSTEM_PROCESSOR=" ${CMAKE_HOST_SYSTEM_PROCESSOR})

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(LIBLLM_KERNEL_X86_64 ON)
endif()
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(LIBLLM_KERNEL_AARCH64 ON)
endif()

check_type_size(__fp16 FP16 BUILTIN_TYPES_ONLY LANGUAGE CXX)
if(HAVE_FP16)
    message("__fp16 exists.")
    add_compile_definitions(LIBLLM_HAVE_FP16)
endif()

add_subdirectory("cpp")

set(libllm_LIBADD lutil ${CMAKE_DL_LIBS})

if(WITH_OPENMP)
    set(libllm_LIBADD ${libllm_LIBADD} OpenMP::OpenMP_CXX)
endif()

add_library(libllm SHARED $<TARGET_OBJECTS:libllm_static>)
target_link_libraries(libllm ${libllm_LIBADD}  )
set_property(TARGET libllm PROPERTY OUTPUT_NAME llm)
if(UNIX AND NOT APPLE)
    target_link_options(libllm PUBLIC "-Wl,--no-undefined")
endif()

set(unittest_LIBADD
    libllm_static
    libllm_test
    lutil_test
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

add_library(catch2 STATIC  "../third_party/catch2/catch_amalgamated.cpp")

add_executable(unittest "cpp/libllm/test_main.cc")
target_include_directories(unittest PRIVATE "cpp")

target_link_libraries(unittest ${unittest_LIBADD})

if (WITH_CUDA)
    add_library(llmplugincublas SHARED $<TARGET_OBJECTS:llmplugincublas_static>)
    target_include_directories(llmplugincublas PRIVATE ${libllm_INCDIR})
    target_link_libraries(llmplugincublas lutil CUDA::cublas)
    if(UNIX AND NOT APPLE)
        target_link_options(llmplugincublas PUBLIC "-Wl,--no-undefined")
    endif()
endif()

if (WITH_FFMPEG)
    add_library(llmpluginffmpeg SHARED $<TARGET_OBJECTS:llmpluginffmpeg_static>)
    target_include_directories(llmpluginffmpeg PRIVATE ${libllm_INCDIR})

    set(ffmpeg_ROOT "third_party/ffmpeg")
    target_link_directories(llmpluginffmpeg PUBLIC
        "${ffmpeg_ROOT}/libavformat"
        "${ffmpeg_ROOT}/libavcodec"
        "${ffmpeg_ROOT}/libavfilter"
        "${ffmpeg_ROOT}/libavutil"
        "${ffmpeg_ROOT}/libswresample")
    target_link_libraries(
        llmpluginffmpeg avformat avcodec avfilter avutil swresample ${CMAKE_THREAD_LIBS_INIT}) 
    if(UNIX AND NOT APPLE)
        target_link_options(llmpluginffmpeg PUBLIC "-Wl,--no-undefined")
    endif()
endif()

enable_testing()
add_test(NAME unittest COMMAND $<TARGET_FILE:unittest>)
