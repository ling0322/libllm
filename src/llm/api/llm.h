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

#pragma once

#include <stdint.h>

#define LLMAPI

typedef int32_t LIBLLM_STATUS;
typedef int32_t LIBLLM_BOOL;

#define LIBLLM_TRUE 1
#define LIBLLM_FALSE 0
#define LIBLLM_OK 0

#define LIBLLM_DEVICE_CPU  0x0000
#define LIBLLM_DEVICE_CUDA 0x0100
#define LIBLLM_DEVICE_AUTO 0x1f00

typedef struct llm_model_t llm_model_t;
typedef struct llm_compl_opt_t llm_compl_opt_t;
typedef struct llm_compl_t llm_compl_t;
typedef struct llm_chunk_t llm_chunk_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LLMAPI LIBLLM_STATUS llm_init();
LLMAPI void llm_destroy();

LLMAPI llm_model_t *llm_model_init(const char *ini_path);
LLMAPI void llm_model_destroy(llm_model_t *m);
LLMAPI const char *llm_model_get_name(llm_model_t *m);
LLMAPI llm_compl_t *llm_model_complete(llm_model_t *m, llm_compl_opt_t *o);

LLMAPI llm_compl_opt_t *llm_compl_opt_init();
LLMAPI void llm_compl_opt_destroy(llm_compl_opt_t *o);
LLMAPI LIBLLM_STATUS llm_compl_opt_set_top_p(llm_compl_opt_t *o, float topp);
LLMAPI LIBLLM_STATUS llm_compl_opt_set_temperature(llm_compl_opt_t *o, float temperature);
LLMAPI LIBLLM_STATUS llm_compl_opt_set_prompt(llm_compl_opt_t *o, const char *prompt);
LLMAPI LIBLLM_STATUS llm_compl_opt_set_top_k(llm_compl_opt_t *o, int32_t topk);

LLMAPI void llm_compl_destroy(llm_compl_t *c);
LLMAPI LIBLLM_BOOL llm_compl_is_active(llm_compl_t *c);
LLMAPI llm_chunk_t *llm_compl_next_chunk(llm_compl_t *c);

LLMAPI const char *llm_chunk_get_text(llm_chunk_t *c);
LLMAPI void llm_chunk_destroy(llm_chunk_t *c);

LLMAPI const char *llm_get_last_error_message();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
