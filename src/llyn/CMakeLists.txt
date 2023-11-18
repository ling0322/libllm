find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(llyn_SOURCES
    "cpu/apply_rotary_pos_emb.cc"
    "cpu/attention.cc"
    "cpu/cat.cc"
    "cpu/copy.cc"
    "cpu/cpu_operators.cc"
    "cpu/internal.cc"
    "cpu/lookup.cc"
    "cpu/matmul.cc"
    "cpu/mul.cc"
    "cpu/print.cc"
    "cpu/swiglu.cc"
    "cpu/tensor.cc"
    "cpu/view.cc"
    "cuda/cuda_common.cc"
    "cuda/cuda_operators.cc"
    "cuda/to_device.cc"
    "internal/cpu_tensor_data.cc"
    "internal/cuda_tensor_data.cc"
    "internal/operators.cc"
    "internal/tensor_data.cc"
    "internal/tensor_shape.cc"
    "nn/embedding.cc"
    "nn/layer_norm.cc"
    "nn/linear.cc"
    "nn/rms_norm.cc"
    "context.cc"
    "device.cc"
    "dtype.cc"
    "functional.cc"
    "llyn.cc"
    "state_map.cc"
    "tensor.cc"
    "util.cc")

set(llyn_INCDIR ..)

if(CUDAToolkit_FOUND)
    set(llyn_INCDIR "${llyn_INCDIR}" ${CUDAToolkit_INCLUDE_DIRS})
endif()

add_library(llyn STATIC ${llyn_SOURCES} )
target_include_directories(llyn PRIVATE ${llyn_INCDIR})
target_compile_features(llyn PRIVATE cxx_std_14)
target_link_libraries(llyn lymath lyutil)
target_link_options(llyn PUBLIC "-fopenmp")

set(llyn_cuda_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/cuda/test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/../../third_party/catch2/catch_amalgamated.cpp")
add_executable(llyn_cuda_test
    ${llyn_cuda_test_SOURCES})
target_include_directories(llyn_cuda_test PRIVATE .. ../.. ../third_party)
