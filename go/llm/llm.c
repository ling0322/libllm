// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

// global state
void (*p_llm_init)();
const char *(*p_llm_get_last_error_message)();

// llm

int32_t (*p_llm_model_init)(llm_model_t *m);
int32_t (*p_llm_model_destroy)(llm_model_t *m);
int32_t (*p_llm_model_load)(llm_model_t *m, llm_json_t *kwargs);
int32_t (*p_llm_model_get_info)(llm_model_t *m, llm_json_t *info);
int32_t (*p_llm_model_complete)(llm_model_t *m, llm_json_t *kwargs, llm_completion_t *comp);

int32_t (*p_llm_completion_init)(llm_completion_t *c);
int32_t (*p_llm_completion_destroy)(llm_completion_t *c);
int32_t (*p_llm_completion_get_next_chunk)(llm_completion_t *c, llm_json_t *chunk);

// json
int32_t (*p_llm_json_init)(llm_json_t *j);
int32_t (*p_llm_json_destroy)(llm_json_t *j);
int32_t (*p_llm_json_parse)(llm_json_t *j, const char *json_str);
int32_t (*p_llm_json_dump)(llm_json_t *j, char *buf, int64_t buf_size);

// asr

int32_t (*p_llm_asr_model_init)(llm_asr_model_t *m);
int32_t (*p_llm_asr_model_load)(llm_asr_model_t *m, llm_json_t *options);
int32_t (*p_llm_asr_model_destroy)(llm_asr_model_t *m);
int32_t (*p_llm_asr_recognition_init)(llm_asr_recognition_t *r);
int32_t (*p_llm_asr_recognition_destroy)(llm_asr_recognition_t *r);
int32_t (*p_llm_asr_recognition_get_next_result)(llm_asr_recognition_t *r, llm_json_t *result);
int32_t (*p_llm_asr_recognize_media_file)(
    llm_asr_model_t *model,
    llm_json_t *options,
    llm_asr_recognition_t *recognition);

// load the libllm shared library.
void *llm_load_library(const char *library_path) {
  // first try to load the dll from same folder as current module.
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

#define LOAD_SYMBOL(hDll, symbol)                                    \
  p_##symbol = GET_PROC_ADDR(hDll, #symbol);                         \
  if (!p_##symbol) {                                                 \
    fprintf(stderr, "llm.go: unable to load symbol: %s\n", #symbol); \
    return LLM_ERROR_ABORTED;                                        \
  }

int32_t llm_load_symbols(void *pDll) {
  LLM_HMODULE hDll = (LLM_HMODULE)pDll;

  LOAD_SYMBOL(hDll, llm_init);
  LOAD_SYMBOL(hDll, llm_get_last_error_message);
  LOAD_SYMBOL(hDll, llm_model_init);
  LOAD_SYMBOL(hDll, llm_model_destroy);
  LOAD_SYMBOL(hDll, llm_model_load);
  LOAD_SYMBOL(hDll, llm_model_get_info);
  LOAD_SYMBOL(hDll, llm_model_complete);
  LOAD_SYMBOL(hDll, llm_completion_init);
  LOAD_SYMBOL(hDll, llm_completion_destroy);
  LOAD_SYMBOL(hDll, llm_completion_get_next_chunk);
  LOAD_SYMBOL(hDll, llm_json_init);
  LOAD_SYMBOL(hDll, llm_json_destroy);
  LOAD_SYMBOL(hDll, llm_json_parse);
  LOAD_SYMBOL(hDll, llm_json_dump);
  LOAD_SYMBOL(hDll, llm_asr_model_init);
  LOAD_SYMBOL(hDll, llm_asr_model_load);
  LOAD_SYMBOL(hDll, llm_asr_model_destroy);
  LOAD_SYMBOL(hDll, llm_asr_recognition_init);
  LOAD_SYMBOL(hDll, llm_asr_recognition_destroy);
  LOAD_SYMBOL(hDll, llm_asr_recognition_get_next_result);
  LOAD_SYMBOL(hDll, llm_asr_recognize_media_file);

  return 0;
}

void llm_init() {
  return p_llm_init();
}

const char *llm_get_last_error_message() {
  return p_llm_get_last_error_message();
}

int32_t llm_model_init(llm_model_t *m) {
  return p_llm_model_init(m);
}

int32_t llm_model_destroy(llm_model_t *m) {
  return p_llm_model_destroy(m);
}

int32_t llm_model_load(llm_model_t *m, llm_json_t *kwargs) {
  return p_llm_model_load(m, kwargs);
}

int32_t llm_model_get_info(llm_model_t *m, llm_json_t *info) {
  return p_llm_model_get_info(m, info);
}

int32_t llm_model_complete(llm_model_t *m, llm_json_t *kwargs, llm_completion_t *comp) {
  return p_llm_model_complete(m, kwargs, comp);
}

int32_t llm_completion_init(llm_completion_t *c) {
  return p_llm_completion_init(c);
}

int32_t llm_completion_destroy(llm_completion_t *c) {
  return p_llm_completion_destroy(c);
}

int32_t llm_completion_get_next_chunk(llm_completion_t *c, llm_json_t *chunk) {
  return p_llm_completion_get_next_chunk(c, chunk);
}

int32_t llm_json_init(llm_json_t *j) {
  return p_llm_json_init(j);
}

int32_t llm_json_destroy(llm_json_t *j) {
  return p_llm_json_destroy(j);
}

int32_t llm_json_parse(llm_json_t *j, const char *json_str) {
  return p_llm_json_parse(j, json_str);
}

int32_t llm_json_dump(llm_json_t *j, char *buf, int64_t buf_size) {
  return p_llm_json_dump(j, buf, buf_size);
}

int32_t llm_asr_model_init(llm_asr_model_t *m) {
  return p_llm_asr_model_init(m);
}

int32_t llm_asr_model_load(llm_asr_model_t *m, llm_json_t *options) {
  return p_llm_asr_model_load(m, options);
}

int32_t llm_asr_model_destroy(llm_asr_model_t *m) {
  return p_llm_asr_model_destroy(m);
}

int32_t llm_asr_recognition_init(llm_asr_recognition_t *r) {
  return p_llm_asr_recognition_init(r);
}

int32_t llm_asr_recognition_destroy(llm_asr_recognition_t *r) {
  return p_llm_asr_recognition_destroy(r);
}

int32_t llm_asr_recognition_get_next_result(llm_asr_recognition_t *r, llm_json_t *result) {
  return p_llm_asr_recognition_get_next_result(r, result);
}

int32_t llm_asr_recognize_media_file(
    llm_asr_model_t *model,
    llm_json_t *options,
    llm_asr_recognition_t *recognition) {
  return p_llm_asr_recognize_media_file(model, options, recognition);
}
