set(lytok_SOURCES
    "bpe_config.cc"
    "bpe_encoder.cc"
    "bpe_model.cc"
    "tokenizer.cc")

add_library(lytok STATIC ${lytok_SOURCES})
target_include_directories(lytok PRIVATE .. ../..)

