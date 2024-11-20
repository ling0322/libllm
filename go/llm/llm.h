// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LIBLLM_LLM_API_
#define LIBLLM_LLM_API_

#include <stdint.h>

#define LLM_ERROR_INVALID_ARG 0x0100
#define LLM_ERROR_INSUFFICIENT_BUFFER 0x0101
#define LLM_ERROR_ABORTED 0x0102
#define LLM_ERROR_EOF 0x0103

typedef struct llm_model_impl_t *llm_model_t;
typedef struct llm_completion_impl_t *llm_completion_t;
typedef struct llm_json_impl_t *llm_json_t;
typedef struct llm_asr_recognition_impl_t *llm_asr_recognition_t;
typedef struct llm_asr_model_impl_t *llm_asr_model_t;

// library

void *llm_load_library(const char *library_path);
int32_t llm_load_symbols(void *pDll);

// global state
void llm_init();
const char *llm_get_last_error_message();

// JSON

int32_t llm_json_init(llm_json_t *j);
int32_t llm_json_destroy(llm_json_t *j);
int32_t llm_json_parse(llm_json_t *j, const char *json_str);
int32_t llm_json_dump(llm_json_t *j, char *buf, int64_t buf_size);

// LLM

int32_t llm_model_init(llm_model_t *m);
int32_t llm_model_destroy(llm_model_t *m);
int32_t llm_model_load(llm_model_t *m, llm_json_t *kwargs);
int32_t llm_model_get_info(llm_model_t *m, llm_json_t *info);
int32_t llm_model_complete(llm_model_t *m, llm_json_t *kwargs, llm_completion_t *comp);

int32_t llm_completion_init(llm_completion_t *c);
int32_t llm_completion_destroy(llm_completion_t *c);
int32_t llm_completion_get_next_chunk(llm_completion_t *c, llm_json_t *chunk);

// JSON

int32_t llm_json_init(llm_json_t *j);
int32_t llm_json_destroy(llm_json_t *j);
int32_t llm_json_parse(llm_json_t *j, const char *json_str);
int32_t llm_json_dump(llm_json_t *j, char *buf, int64_t buf_size);

// ASR

int32_t llm_asr_model_init(llm_asr_model_t *m);
int32_t llm_asr_model_load(llm_asr_model_t *m, llm_json_t *options);
int32_t llm_asr_model_destroy(llm_asr_model_t *m);

int32_t llm_asr_recognition_init(llm_asr_recognition_t *r);
int32_t llm_asr_recognition_destroy(llm_asr_recognition_t *r);
int32_t llm_asr_recognition_get_next_result(llm_asr_recognition_t *r, llm_json_t *result);

int32_t llm_asr_recognize_media_file(
    llm_asr_model_t *model,
    llm_json_t *options,
    llm_asr_recognition_t *recognition);

#endif  // LIBLLM_LLM_API_
