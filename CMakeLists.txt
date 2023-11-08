cmake_minimum_required(VERSION 3.11)
project(llmpp LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-O3 -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MKL_PREFIX "/opt/intel/mkl" CACHE STRING "Prefix for MKL headers and libraries.")

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

if(WIN32)
  add_definitions( "/D_CRT_SECURE_NO_WARNINGS /wd4267 /DCATCH_AMALGAMATED_CUSTOM_MAIN" )
endif(WIN32)
if(UNIX)
  add_definitions( "-DCATCH_AMALGAMATED_CUSTOM_MAIN -DCATCH_CONFIG_PREFIX_ALL" )
endif(UNIX)

add_subdirectory("src")
