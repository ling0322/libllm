cmake_minimum_required(VERSION 3.11)
project(llmpp LANGUAGES CXX)

if(UNIX)
  
endif(UNIX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MKL_PREFIX "/opt/intel/mkl" CACHE STRING "Prefix for MKL headers and libraries.")

if(WIN32)
  add_definitions( "/D_CRT_SECURE_NO_WARNINGS /wd4267 /DCATCH_AMALGAMATED_CUSTOM_MAIN" )
endif(WIN32)
if(UNIX)
  add_definitions( "-DCATCH_AMALGAMATED_CUSTOM_MAIN -DCATCH_CONFIG_PREFIX_ALL" )
  set(CMAKE_CXX_FLAGS "-O3 -g")
endif(UNIX)

add_subdirectory("src")
