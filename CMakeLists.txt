cmake_minimum_required(VERSION 3.25)


project(libllm LANGUAGES CXX)


find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS}) 

set(CMAKE_CUDA_ARCHITECTURES 70 75)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(MKL_PREFIX "/opt/intel/mkl" CACHE STRING "Prefix for MKL headers and libraries.")

set (CUDA_ENABLED ON)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

if(WIN32)
  add_definitions( "/D_CRT_SECURE_NO_WARNINGS /wd4267 /DCATCH_AMALGAMATED_CUSTOM_MAIN" )
endif(WIN32)
if(UNIX)
  add_definitions( "-DCATCH_AMALGAMATED_CUSTOM_MAIN -DCATCH_CONFIG_PREFIX_ALL" )
endif(UNIX)

add_subdirectory("src")
