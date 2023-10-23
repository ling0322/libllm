find_package(OpenMP REQUIRED)

if(WIN32)
    set_source_files_properties("kernel_avx512.cc" PROPERTIES COMPILE_FLAGS /arch:AVX512)
    set_source_files_properties("kernel_avx2.cc" PROPERTIES COMPILE_FLAGS /arch:AVX2)
endif(WIN32)
if(UNIX)
    set_source_files_properties("kernel_avx512.cc" PROPERTIES COMPILE_FLAGS "-mavx512f")
    set_source_files_properties("kernel_avx2.cc" PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
endif(UNIX)

set(lymath_SOURCES
    "kernel_avx512.cc"
    "kernel_avx2.cc"
    "kernel_fallback.cc"
    "lymath.cc"
    "util.cc")

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(lymath STATIC ${lymath_SOURCES} )
target_include_directories(lymath PRIVATE .. ../..)
target_compile_features(lymath PRIVATE cxx_std_14)

set(lymath_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/benchmark.cc"
    "${CMAKE_CURRENT_LIST_DIR}/../../third_party/catch2/catch_amalgamated.cpp")

find_library(MKL_GF_LP64_LIBRARY NAMES mkl_gf_lp64 HINTS ${MKL_PREFIX}/lib/intel64)
find_library(MKL_CORE_LIBRARY NAMES mkl_core HINTS ${MKL_PREFIX}/lib/intel64)
find_library(MKL_GNU_THREAD_LIBRARY NAMES mkl_gnu_thread HINTS ${MKL_PREFIX}/lib/intel64)
    

add_executable(lymath_test
    ${lymath_test_SOURCES})

target_include_directories(lymath_test PRIVATE .. ../.. ../third_party ${MKL_PREFIX}/include)
target_link_libraries(lymath_test
    lymath
    lyutil
    OpenMP::OpenMP_CXX
    ${MKL_GF_LP64_LIBRARY}
    ${MKL_CORE_LIBRARY}
    ${MKL_GNU_THREAD_LIBRARY})
target_link_options(lymath_test PUBLIC "-fopenmp")
