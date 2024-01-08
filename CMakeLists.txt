cmake_minimum_required(VERSION 3.22)
project(libllm LANGUAGES CXX)

if(CUDA_ENABLED)
    add_definitions("-DLLYN_CUDA_ENABLED")
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS}) 
    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75 80 86)
    enable_language(CUDA)
    if(WIN32)
        set(cuda_LIBRARIES CUDA::cudart_static)
    endif(WIN32)
    if(UNIX)
        set(cuda_LIBRARIES CUDA::cudart_static)
    endif(UNIX)
endif(CUDA_ENABLED)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(MKL_PREFIX "/opt/intel/mkl" CACHE STRING "Prefix for MKL headers and libraries.")

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

if(WIN32)
  add_definitions( "/D_CRT_SECURE_NO_WARNINGS /DCATCH_AMALGAMATED_CUSTOM_MAIN /DCATCH_CONFIG_PREFIX_ALL" )
endif(WIN32)
if(UNIX)
  add_definitions( "-DCATCH_AMALGAMATED_CUSTOM_MAIN -DCATCH_CONFIG_PREFIX_ALL" )
  set(CMAKE_CXX_FLAGS "-O3 -g")
endif(UNIX)

add_subdirectory("src")
