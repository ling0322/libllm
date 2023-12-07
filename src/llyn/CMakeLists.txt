find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(llyn_SOURCES
    "operators/operators.cc"
    "operators/common/common.cc"
    "operators/common/matmul.cc"
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
    "llyn.cc"
    "state_map.cc"
    "tensor.cc")

add_library(llyn STATIC ${llyn_SOURCES} )
target_include_directories(llyn PRIVATE ..)
target_compile_features(llyn PRIVATE cxx_std_14)
target_link_libraries(llyn lymath lyutil)
if(WIN32)
  target_compile_definitions(llyn PUBLIC  "/openmp")
endif(WIN32)
if(UNIX)
  target_link_options(llyn PUBLIC "-fopenmp")
endif(UNIX)

if (CUDA_ENABLED)
    set(cudnn_LIBRARIES "${CUDNN_ROOT}/lib/libcudnn_ops_infer_static.a")
    set(llynextcuda_INCDIR "${llyn_INCDIR}" ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_ROOT}/include)
    set(llynextcuda_SOURCES
        "operators/cuda/apply_rotary_pos_emb.cu"
        "operators/cuda/cast.cu"
        "operators/cuda/causal_mask.cu"
        "operators/cuda/copy.cu"
        "operators/cuda/common.cc"
        "operators/cuda/cuda_operators.cc"
        "operators/cuda/cuda_tensor_data.cc"
        "operators/cuda/cudnn_wrapper.cc"
        "operators/cuda/dequant.cu"
        "operators/cuda/lookup.cu"
        "operators/cuda/matmul.cc"
        "operators/cuda/matvec.cu"
        "operators/cuda/rms_norm.cu"
        "operators/cuda/swiglu.cu"
        "operators/cuda/to_device.cc")
    add_library(llynextcuda STATIC ${llynextcuda_SOURCES})
    target_include_directories(llynextcuda PRIVATE .. ${llynextcuda_INCDIR})
    target_compile_features(llynextcuda PRIVATE cxx_std_14)

    set(llyn_test_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/operators/cuda/test.cc"
        PARENT_SCOPE)
endif()