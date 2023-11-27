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

add_library(llyn STATIC ${llyn_SOURCES} )
target_include_directories(llyn PRIVATE .. ${MKL_PREFIX}/include)
target_compile_features(llyn PRIVATE cxx_std_14)
target_link_libraries(llyn lymath lyutil)
if(WIN32)
  target_compile_definitions(llyn PUBLIC  "/openmp")
endif(WIN32)
if(UNIX)
  target_link_options(llyn PUBLIC "-fopenmp")
endif(UNIX)

set(llyn_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/internal/operators_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/nn/rms_norm_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/tensor_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/nn_test.cc"
    PARENT_SCOPE)
