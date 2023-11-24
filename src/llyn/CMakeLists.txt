find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(llyn_SOURCES
    "operators/operators.cc"
    "operators/cpu/apply_rotary_pos_emb.cc"
    "operators/cpu/attention.cc"
    "operators/cpu/cat.cc"
    "operators/cpu/copy.cc"
    "operators/cpu/cpu_operators.cc"
    "operators/cpu/cpu_tensor_data.cc"
    "operators/cpu/lookup.cc"
    "operators/cpu/matmul.cc"
    "operators/cpu/mul.cc"
    "operators/cpu/print.cc"
    "operators/cpu/swiglu.cc"
    "operators/cpu/tensor.cc"
    "operators/cpu/view.cc"
    "operators/cuda/cast.cu"
    "operators/cuda/copy.cu"
    "operators/cuda/create_tensor.cc"
    "operators/cuda/cuda_common.cc"
    "operators/cuda/cuda_operators.cc"
    "operators/cuda/cuda_tensor_data.cc"
    "operators/cuda/cudnn_operators.cc"
    "operators/cuda/lookup.cu"
    "operators/cuda/to_device.cc"
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
    "tensor.cc")

set(llyn_INCDIR ..)

if(CUDAToolkit_FOUND)
  set(llyn_INCDIR "${llyn_INCDIR}" ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_ROOT}/include)
endif()

add_library(llyn STATIC ${llyn_SOURCES} )
target_include_directories(llyn PRIVATE ${llyn_INCDIR})
target_compile_features(llyn PRIVATE cxx_std_14)
target_link_libraries(llyn lymath lyutil)
target_link_options(llyn PUBLIC "-fopenmp")

set(cudnn_LIBRARIES
    "${CUDNN_ROOT}/lib/libcudnn_ops_infer_static.a")

set(llyn_cuda_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/operators/cuda/test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/../../third_party/catch2/catch_amalgamated.cpp")
add_executable(llyn_cuda_test
    ${llyn_cuda_test_SOURCES})
target_link_libraries(llyn_cuda_test llyn lymath lyutil ${cudnn_LIBRARIES} CUDA::cudart_static CUDA::cublas_static)
target_include_directories(llyn_cuda_test PRIVATE .. ../.. ../third_party)
