// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <stdint.h>

#define LLM_ERROR_INVALID_ARG 0x0100
#define LLM_ERROR_ABORTED 0x0102

typedef struct llm_engine_impl_t *llm_engine_t;

typedef struct llm_string_view_t {
  const char *data;
  int64_t size;
} llm_string_view_t;

typedef enum llm_device_type_t {
  LLM_DEVICE_AUTO = 0,
  LLM_DEVICE_CPU = 1,
  LLM_DEVICE_CUDA = 2,
} llm_device_type_t;

typedef struct llm_engine_options_t {
  int64_t struct_size;
  llm_string_view_t model_path;
  llm_device_type_t device;
  int32_t max_num_batched_tokens;
  int32_t kv_cache_block_size;
  float kv_cache_memory_utilization;
} llm_engine_options_t;

typedef struct llm_generation_config_t {
  int64_t struct_size;
  int32_t top_k;
  float top_p;
  float temperature;
  int32_t max_tokens;
} llm_generation_config_t;

typedef struct llm_message_t {
  llm_string_view_t role;
  llm_string_view_t content;
} llm_message_t;

typedef struct llm_request_t {
  int64_t struct_size;
  llm_string_view_t request_id;
  const int64_t *input_ids;
  int64_t num_input_ids;
  const llm_message_t *messages;
  int64_t num_messages;
  llm_generation_config_t generation_config;
} llm_request_t;

typedef enum llm_finish_reason_t {
  LLM_FINISH_REASON_NONE = 0,
  LLM_FINISH_REASON_STOP = 1,
  LLM_FINISH_REASON_LENGTH = 2,
  LLM_FINISH_REASON_CANCELLED = 3,
  LLM_FINISH_REASON_ERROR = 4,
} llm_finish_reason_t;

typedef struct llm_request_output_t {
  llm_string_view_t request_id;
  const int64_t *token_ids;
  int64_t num_token_ids;
  llm_string_view_t text;
  int32_t finished;
  llm_finish_reason_t finish_reason;
  llm_string_view_t error_message;
} llm_request_output_t;

typedef struct llm_request_outputs_t {
  const llm_request_output_t *data;
  int64_t size;
} llm_request_outputs_t;

typedef void (*llm_stream_callback_t)(const llm_request_outputs_t *outputs, void *user_data);

void *llm_load_library(const char *library_path);
int32_t llm_load_symbols(void *library);

void llm_init(void);
int32_t llm_get_last_error_code(void);
const char *llm_get_last_error_message(void);
void llm_engine_options_init(llm_engine_options_t *options);
void llm_generation_config_init(llm_generation_config_t *config);
void llm_request_init(llm_request_t *request);
int32_t llm_engine_init(llm_engine_t *engine);
int32_t llm_engine_load_go(llm_engine_t *engine, const llm_engine_options_t *options);
int32_t llm_engine_destroy(llm_engine_t *engine);
int32_t llm_engine_add_request(llm_engine_t *engine, const llm_request_t *request);
int32_t llm_engine_abort_request(llm_engine_t *engine, llm_string_view_t request_id);
