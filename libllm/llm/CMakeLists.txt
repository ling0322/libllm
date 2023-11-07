find_package(OpenMP REQUIRED)

set(libllm_INCDIRS .. ../../third_party)
set(libllm_SOURCES
    "api/libllm.cc"
    "api/environment.cc"
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
    lyutil)

add_library(llm STATIC ${libllm_SOURCES} )
target_include_directories(llm PRIVATE ${libllm_INCDIRS})
target_compile_features(llm PRIVATE cxx_std_14)
target_link_libraries(llm ${libllm_LIBADD}  )

set(llm_SOURCES
    "cli/dialog_manager.cc"
    "cli/llm.cc")

add_executable(llm_main ${llm_SOURCES})
target_include_directories(llm_main PRIVATE ${libllm_INCDIRS})
target_link_libraries(llm_main llm OpenMP::OpenMP_CXX)
target_link_options(llm_main PUBLIC "-fopenmp")
set_property(TARGET llm_main PROPERTY OUTPUT_NAME llm)


set(llm_test_SOURCES
    "chatglm2/chatglm2_model_test.cc"
    "cli/test_main.cc"
    "../../third_party/catch2/catch_amalgamated.cpp")

add_executable(llm_test
    ${llm_test_SOURCES})
target_include_directories(llm_test PRIVATE ${libllm_INCDIRS})
target_link_libraries(llm_test llm OpenMP::OpenMP_CXX)
target_link_options(llm_test PUBLIC "-fopenmp")
