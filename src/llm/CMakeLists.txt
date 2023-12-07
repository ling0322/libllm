find_package(OpenMP REQUIRED)

set(libllm_INCDIRS .. ../../third_party)
set(libllm_SOURCES
    "api/llm.cc"
    "api/model_factory.cc"
    "chatglm2/chatglm2_config.cc"
    "chatglm2/chatglm2_model.cc"
    "chatglm2/chatglm2_model_for_generation.cc"
    "chatglm2/glm_block.cc"
    "chatglm2/mlp.cc"
    "chatglm2/self_attention.cc"
    "common/generator.cc"
    "common/sampler.cc"
    "llama/attention.cc"
    "llama/decoder_layer.cc"
    "llama/llama_config.cc"
    "llama/llama_model.cc"
    "llama/llama_model_for_generation.cc"
    "llama/mlp.cc")

if(WIN32)
    set(cuda_LIBRARIES CUDA::cudart CUDA::cublas)
endif(WIN32)
if(UNIX)
    set(cuda_LIBRARIES CUDA::cudart_static CUDA::cublas_static)
endif(UNIX)

set(libllm_LIBADD
    llyn
    llynextcuda
    lymath
    lytok
    lyutil
    OpenMP::OpenMP_CXX
    ${cudnn_LIBRARIES}
    ${cuda_LIBRARIES})

add_library(libllm SHARED ${libllm_SOURCES} )
target_include_directories(libllm PRIVATE ${libllm_INCDIRS})
target_compile_features(libllm PRIVATE cxx_std_14)
target_link_directories(libllm PUBLIC "${CUDNN_ROOT}/lib/" "${CUDNN_ROOT}/lib/x64")
target_link_libraries(libllm ${libllm_LIBADD}  )
set_property(TARGET libllm PROPERTY OUTPUT_NAME llm)
target_link_options(libllm PUBLIC "-Wl,--no-undefined")

set(llm_SOURCES
    "cli/dialog_manager.cc"
    "cli/llm.cc")

add_executable(llm ${llm_SOURCES})
target_include_directories(llm PRIVATE ${libllm_INCDIRS})
target_link_libraries(llm libllm OpenMP::OpenMP_CXX )
target_link_options(llm PUBLIC "-fopenmp")


set(llm_test_SOURCES
    ${libllm_SOURCES}
    ${lymath_test_SOURCES}
    ${llyn_test_SOURCES}
    "cli/test_main.cc"
    "../../third_party/catch2/catch_amalgamated.cpp")

add_executable(unittest ${llm_test_SOURCES})
target_include_directories(unittest PRIVATE ${libllm_INCDIRS})
target_link_directories(unittest PUBLIC "${CUDNN_ROOT}/lib/" "${CUDNN_ROOT}/lib/x64")
target_link_libraries(unittest ${libllm_LIBADD})
target_link_options(unittest PUBLIC "-fopenmp")
