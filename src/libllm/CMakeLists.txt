if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set(libllm_SOURCES
    "bpe.cc"
    "generator.cc"
    "kv_cache.cc"
    "layers.cc"
    "llama.cc"
    "llm.cc"
    "model_for_generation.cc"
    "prompt.cc"
    "tokenizer.cc"
    "var_builder.cc")

set(libllm_INCDIR ".." "../../third_party")

add_library(libllm_static OBJECT ${libllm_SOURCES})
target_compile_options(libllm_static PRIVATE "-DLIBLLM_EXPORTS")
target_include_directories(libllm_static PRIVATE ${LIBDWARF_INCLUDE_DIRS})
#set_target_properties(libllm_static PROPERTIES CXX_VISIBILITY_PRESET hidden)
#set_target_properties(libllm_static PROPERTIES CUDA_VISIBILITY_PRESET hidden)
target_include_directories(libllm_static PRIVATE ${libllm_INCDIR})
