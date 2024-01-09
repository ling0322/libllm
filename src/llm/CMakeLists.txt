find_package(OpenMP REQUIRED)

set(libllm_INCDIRS .. ../../third_party)
set(libllm_SOURCES
    "api/llm.cc"
    "api/model_factory.cc"
    "chatglm/chatglm_config.cc"
    "chatglm/chatglm_model.cc"
    "chatglm/chatglm_model_for_generation.cc"
    "chatglm/glm_block.cc"
    "chatglm/mlp.cc"
    "chatglm/self_attention.cc"
    "common/generator.cc"
    "common/sampler.cc"
    "llama/attention.cc"
    "llama/decoder_layer.cc"
    "llama/llama_config.cc"
    "llama/llama_model.cc"
    "llama/llama_model_for_generation.cc"
    "llama/mlp.cc")


set(libllm_LIBADD
    llyn
    lymath
    lytok
    OpenMP::OpenMP_CXX)

if(LLYN_CUDA)
    set(libllm_LIBADD
        ${libllm_LIBADD}
        llyncuda
        ${cuda_LIBRARIES})

    add_library(llyngemmextcublas SHARED ${llyngemmextcublas_SOURCES})
    target_include_directories(llyngemmextcublas PRIVATE ${libllm_INCDIRS})
    target_link_libraries(llyngemmextcublas lyutil CUDA::cublas)
    target_link_options(llyngemmextcublas PUBLIC "-Wl,--no-undefined")
endif()

set(libllm_LIBADD
    ${libllm_LIBADD}
    lyutil
    OpenMP::OpenMP_CXX)

add_library(libllm SHARED ${libllm_SOURCES} )
target_include_directories(libllm PRIVATE ${libllm_INCDIRS})
target_compile_features(libllm PRIVATE cxx_std_14)
target_link_libraries(libllm ${libllm_LIBADD}  )
set_property(TARGET libllm PROPERTY OUTPUT_NAME llm)
target_link_options(libllm PUBLIC "-Wl,--no-undefined")

add_library(libllm_static ${libllm_SOURCES} )
target_include_directories(libllm_static PRIVATE ${libllm_INCDIRS})
target_compile_features(libllm_static PRIVATE cxx_std_14)

set(llm_SOURCES
    "cli/dialog_manager.cc"
    "cli/llm.cc")

add_executable(llm ${llm_SOURCES})
target_include_directories(llm PRIVATE ${libllm_INCDIRS})
target_link_libraries(llm libllm OpenMP::OpenMP_CXX )
target_link_options(llm PUBLIC "-fopenmp")


set(llm_test_SOURCES
    ${lymath_test_SOURCES}
    ${llyn_test_SOURCES}
    "cli/test_main.cc"
    "chatglm/chatglm_model_test.cc"
    "chatglm/glm_block_test.cc"
    "chatglm/mlp_test.cc"
    "chatglm/self_attention_test.cc"
    "../../third_party/catch2/catch_amalgamated.cpp")

add_executable(unittest ${llm_test_SOURCES})
target_include_directories(unittest PRIVATE ${libllm_INCDIRS})
target_link_libraries(unittest libllm_static ${libllm_LIBADD})
target_link_options(unittest PUBLIC "-fopenmp")

enable_testing()
add_test(NAME unittest COMMAND $<TARGET_FILE:unittest>)
