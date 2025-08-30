if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set(libllm_SOURCES
    "bilibili_index.cc"
    "bpe_config.cc"
    "bpe_encoder.cc"
    "bpe_model.cc"
    "generator.cc"
    "llama.cc"
    "llm.cc"
    "model_for_generation.cc"
    "prompt.cc"
    "qwen.cc"
    "tokenizer.cc"
    "wave.cc"
    "wave_stream.cc"
    "whisper.cc"
    "whisper_decoder.cc")

set(libllm_INCDIR ".." "../../third_party")

if (WITH_FFMPEG)
    add_library(llmpluginffmpeg_static OBJECT "read_audio_ffmpeg.cc")
    target_compile_options(llmpluginffmpeg_static PRIVATE "-DLIBLLM_EXPORTS")
    target_include_directories(llmpluginffmpeg_static PRIVATE  ${libllm_INCDIR} "../../third_party/ffmpeg/")
endif()

add_library(libllm_static OBJECT ${libllm_SOURCES})
target_compile_options(libllm_static PRIVATE "-DLIBLLM_EXPORTS")
target_include_directories(libllm_static PRIVATE ${LIBDWARF_INCLUDE_DIRS})
#set_target_properties(libllm_static PROPERTIES CXX_VISIBILITY_PRESET hidden)
#set_target_properties(libllm_static PROPERTIES CUDA_VISIBILITY_PRESET hidden)
target_include_directories(libllm_static PRIVATE ${libllm_INCDIR})
