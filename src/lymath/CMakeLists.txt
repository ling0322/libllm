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

if (ENABLE_MKL)
    add_definitions("-DMKL_ENABLED")
endif()

set(lymath_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/test.cc"
    PARENT_SCOPE)
