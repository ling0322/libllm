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

set(libllm_LIBADD
    OpenMP::OpenMP_CXX
    llyn
    lytok
    lyutil
    CUDA::cudart_static)

add_library(llm-core-static STATIC ${libllm_SOURCES} )
target_include_directories(llm-core-static PRIVATE ${libllm_INCDIRS})
target_compile_features(llm-core-static PRIVATE cxx_std_14)
target_link_libraries(llm-core-static ${libllm_LIBADD}  )

add_library(llm-core MODULE ${libllm_SOURCES} )
target_include_directories(llm-core PRIVATE ${libllm_INCDIRS})
target_compile_features(llm-core PRIVATE cxx_std_14)
target_link_libraries(llm-core ${libllm_LIBADD}  )

set(llm_SOURCES
    "cli/dialog_manager.cc"
    "cli/llm.cc")

add_executable(llm_main ${llm_SOURCES})
target_include_directories(llm_main PRIVATE ${libllm_INCDIRS})
target_link_libraries(llm_main llm-core-static OpenMP::OpenMP_CXX )
target_link_options(llm_main PUBLIC "-fopenmp")
set_property(TARGET llm_main PROPERTY OUTPUT_NAME llm)


set(llm_test_SOURCES
    "chatglm2/chatglm2_model_test.cc"
    "cli/test_main.cc"
    "../../third_party/catch2/catch_amalgamated.cpp")

add_executable(llm_test
    ${llm_test_SOURCES})
target_include_directories(llm_test PRIVATE ${libllm_INCDIRS})
target_link_libraries(llm_test llm-core-static OpenMP::OpenMP_CXX)
target_link_options(llm_test PUBLIC "-fopenmp")
