find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(ly_SOURCES
    "operators/operators.cc"
    "operators/common/common.cc"
    "operators/common/matmul.cc"
    "operators/cpu/apply_rotary_pos_emb.cc"
    "operators/cpu/attention.cc"
    "operators/cpu/cast.cc"
    "operators/cpu/cat.cc"
    "operators/cpu/copy.cc"
    "operators/cpu/cpu_operators.cc"
    "operators/cpu/cpu_tensor_data.cc"
    "operators/cpu/fingerprint.cc"
    "operators/cpu/lookup.cc"
    "operators/cpu/matmul.cc"
    "operators/cpu/mul.cc"
    "operators/cpu/print.cc"
    "operators/cpu/rand.cc"
    "operators/cpu/swiglu.cc"
    "operators/cpu/tensor.cc"
    "operators/cpu/view.cc"
    "internal/tensor_data.cc"
    "internal/tensor_shape.cc"
    "nn/embedding.cc"
    "nn/layer_norm.cc"
    "nn/linear.cc"
    "nn/module.cc"
    "nn/rms_norm.cc"
    "context.cc"
    "device.cc"
    "dtype.cc"
    "functional.cc"
    "ly.cc"
    "state_map.cc"
    "tensor.cc")

add_library(ly STATIC ${ly_SOURCES} )
target_include_directories(ly PRIVATE ..)
target_compile_features(ly PRIVATE cxx_std_14)
target_link_libraries(ly lymath lyutil)
if(WIN32)
  target_compile_definitions(ly PUBLIC  "/openmp")
endif(WIN32)
if(UNIX)
  target_link_options(ly PUBLIC "-fopenmp")
endif(UNIX)

set(ly_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/operators/cpu/test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/nn/test_helper.cc"
    "${CMAKE_CURRENT_LIST_DIR}/nn/embedding_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/nn/linear_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/nn/rms_norm_test.cc")

if (CUDA_ENABLED)
    set(lyextcuda_INCDIR "${llyn_INCDIR}" ${CUDAToolkit_INCLUDE_DIRS})
    set(lyextcuda_SOURCES
        "operators/cuda/apply_rotary_pos_emb.cu"
        "operators/cuda/binary_op.cu"
        "operators/cuda/cast.cu"
        "operators/cuda/causal_mask.cu"
        "operators/cuda/copy.cu"
        "operators/cuda/common.cc"
        "operators/cuda/cuda_operators.cc"
        "operators/cuda/cuda_tensor_data.cc"
        "operators/cuda/dequant.cu"
        "operators/cuda/lookup.cu"
        "operators/cuda/matmul.cc"
        "operators/cuda/matvec.cu"
        "operators/cuda/reduce.cu"
        "operators/cuda/rms_norm.cu"
        "operators/cuda/softmax.cu"
        "operators/cuda/swiglu.cu"
        "operators/cuda/to_device.cc"
        "operators/cuda/transform.cu")
    add_library(lyextcuda STATIC ${lyextcuda_SOURCES})
    target_include_directories(lyextcuda PRIVATE .. ${lyextcuda_INCDIR})
    target_compile_features(lyextcuda PRIVATE cxx_std_14)

    set(lyextgemmcublas_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/operators/cuda/gemm_cublas.cc")
    set(lyextgemmcublas_SOURCES
        ${lyextgemmcublas_SOURCES}
        PARENT_SCOPE)

    set(ly_test_SOURCES
        ${ly_test_SOURCES}
        "${CMAKE_CURRENT_LIST_DIR}/operators/cuda/test.cc")
endif()

set(ly_test_SOURCES
    ${ly_test_SOURCES}
    PARENT_SCOPE)
