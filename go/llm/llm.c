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

#ifdef __APPLE__
#define LUT_PLATFORM_APPLE
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define LUT_PLATFORM_LINUX
#elif defined(WIN32) || defined(__WIN32__) || defined(_MSC_VER) || defined(_WIN32) || \
    defined(__MINGW32__)
#define LUT_PLATFORM_WINDOWS
#else
#error unknown platform
#endif

#if defined(LUT_PLATFORM_APPLE) || defined(LUT_PLATFORM_LINUX)
#include <dlfcn.h>
typedef void *LLM_HMODULE;
#elif defined(LUT_PLATFORM_WINDOWS)
#include <windows.h>
typedef HMODULE LLM_HMODULE;
#endif

#include <stdint.h>
#include <stdio.h>

#include "llm.h"

void (*p_llm_init)(void);
int32_t (*p_llm_get_last_error_code)(void);
const char *(*p_llm_get_last_error_message)(void);
void (*p_llm_engine_options_init)(llm_engine_options_t *options);
void (*p_llm_generation_config_init)(llm_generation_config_t *config);
void (*p_llm_request_init)(llm_request_t *request);
int32_t (*p_llm_engine_init)(llm_engine_t *engine);
int32_t (*p_llm_engine_load)(
    llm_engine_t *engine,
    const llm_engine_options_t *options,
    llm_stream_callback_t callback,
    void *user_data);
int32_t (*p_llm_engine_destroy)(llm_engine_t *engine);
int32_t (*p_llm_engine_add_request)(llm_engine_t *engine, const llm_request_t *request);
int32_t (*p_llm_engine_abort_request)(llm_engine_t *engine, llm_string_view_t request_id);

extern void goLlmStreamCallback(const llm_request_outputs_t *outputs, void *user_data);

void *llm_load_library(const char *library_path) {
#if defined(LUT_PLATFORM_APPLE) || defined(LUT_PLATFORM_LINUX)
  return dlopen(library_path, RTLD_NOW);
#elif defined(LUT_PLATFORM_WINDOWS)
  return LoadLibraryA(library_path);
#endif
}

#if defined(LUT_PLATFORM_APPLE) || defined(LUT_PLATFORM_LINUX)
#define GET_PROC_ADDR dlsym
#elif defined(LUT_PLATFORM_WINDOWS)
#define GET_PROC_ADDR (void *)GetProcAddress
#endif

#define LOAD_SYMBOL(library, symbol)                                  \
  p_##symbol = GET_PROC_ADDR(library, #symbol);                       \
  if (!p_##symbol) {                                                  \
    fprintf(stderr, "llm.go: unable to load symbol: %s\n", #symbol); \
    return LLM_ERROR_ABORTED;                                        \
  }

int32_t llm_load_symbols(void *library) {
  LLM_HMODULE handle = (LLM_HMODULE)library;
  LOAD_SYMBOL(handle, llm_init);
  LOAD_SYMBOL(handle, llm_get_last_error_code);
  LOAD_SYMBOL(handle, llm_get_last_error_message);
  LOAD_SYMBOL(handle, llm_engine_options_init);
  LOAD_SYMBOL(handle, llm_generation_config_init);
  LOAD_SYMBOL(handle, llm_request_init);
  LOAD_SYMBOL(handle, llm_engine_init);
  LOAD_SYMBOL(handle, llm_engine_load);
  LOAD_SYMBOL(handle, llm_engine_destroy);
  LOAD_SYMBOL(handle, llm_engine_add_request);
  LOAD_SYMBOL(handle, llm_engine_abort_request);
  return 0;
}

void llm_init(void) {
  p_llm_init();
}

int32_t llm_get_last_error_code(void) {
  return p_llm_get_last_error_code();
}

const char *llm_get_last_error_message(void) {
  return p_llm_get_last_error_message();
}

void llm_engine_options_init(llm_engine_options_t *options) {
  p_llm_engine_options_init(options);
}

void llm_generation_config_init(llm_generation_config_t *config) {
  p_llm_generation_config_init(config);
}

void llm_request_init(llm_request_t *request) {
  p_llm_request_init(request);
}

int32_t llm_engine_init(llm_engine_t *engine) {
  return p_llm_engine_init(engine);
}

int32_t llm_engine_load_go(llm_engine_t *engine, const llm_engine_options_t *options) {
  return p_llm_engine_load(engine, options, goLlmStreamCallback, NULL);
}

int32_t llm_engine_destroy(llm_engine_t *engine) {
  return p_llm_engine_destroy(engine);
}

int32_t llm_engine_add_request(llm_engine_t *engine, const llm_request_t *request) {
  return p_llm_engine_add_request(engine, request);
}

int32_t llm_engine_abort_request(llm_engine_t *engine, llm_string_view_t request_id) {
  return p_llm_engine_abort_request(engine, request_id);
}
