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

#include "llm_api.h"

// global state
llmStatus_t (*p_llmInit)(int32_t apiVersion);
llmStatus_t (*p_llmDestroy)();
const char *(*p_llmGetLastErrorMessage)();

// llmModel_t
llmModel_t *(*p_llmModel_New)();
llmStatus_t (*p_llmModel_Delete)(llmModel_t *model);
llmStatus_t (*p_llmModel_SetFile)(llmModel_t *model, const char *filename);
llmStatus_t (*p_llmModel_SetDevice)(llmModel_t *model, int32_t device);
llmStatus_t (*p_llmModel_Load)(llmModel_t *model);
const char *(*p_llmModel_GetName)(llmModel_t *model);

// llmPrompt_t
llmPrompt_t *(*p_llmPrompt_New)();
llmStatus_t (*p_llmPrompt_Delete)(llmPrompt_t *prompt);
llmStatus_t (*p_llmPrompt_AppendText)(llmPrompt_t *prompt, const char *text);
llmStatus_t (*p_llmPrompt_AppendControlToken)(llmPrompt_t *prompt, const char *token);
llmStatus_t (*p_llmPrompt_AppendAudio)(
    llmPrompt_t *prompt,
    const llmByte_t *audio,
    int64_t size,
    int32_t format);

// llmCompletion_t
llmCompletion_t *(*p_llmCompletion_New)(llmModel_t *model);
llmStatus_t (*p_llmCompletion_Delete)(llmCompletion_t *comp);
llmStatus_t (*p_llmCompletion_SetPrompt)(llmCompletion_t *comp, llmPrompt_t *prompt);
llmStatus_t (*p_llmCompletion_SetTopP)(llmCompletion_t *comp, float topP);
llmStatus_t (*p_llmCompletion_SetTopK)(llmCompletion_t *comp, int32_t topK);
llmStatus_t (*p_llmCompletion_SetTemperature)(llmCompletion_t *comp, float temperature);
llmStatus_t (*p_llmCompletion_SetConfig)(llmCompletion_t *comp, const char *key, const char *value);
llmBool_t (*p_llmCompletion_Next)(llmCompletion_t *comp);
llmStatus_t (*p_llmCompletion_GetError)(llmCompletion_t *comp);
const char *(*p_llmCompletion_GetText)(llmCompletion_t *comp);
const char *(*p_llmCompletion_GetToken)(llmCompletion_t *comp);

// load the libllm shared library.
LLM_HMODULE llmLoadLibrary(const char *libraryPath) {
  // first try to load the dll from same folder as current module.
#if defined(LUT_PLATFORM_APPLE) || defined(LUT_PLATFORM_LINUX)
  return dlopen(libraryPath, RTLD_NOW);
#elif defined(LUT_PLATFORM_WINDOWS)
  return LoadLibraryA(libraryPath);
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
    return LLM_ABORTED;                                              \
  }

llmStatus_t llmLoadSymbols(LLM_HMODULE hDll) {
  LOAD_SYMBOL(hDll, llmInit);
  LOAD_SYMBOL(hDll, llmDestroy);
  LOAD_SYMBOL(hDll, llmGetLastErrorMessage);
  LOAD_SYMBOL(hDll, llmModel_New);
  LOAD_SYMBOL(hDll, llmModel_Delete);
  LOAD_SYMBOL(hDll, llmModel_SetFile);
  LOAD_SYMBOL(hDll, llmModel_SetDevice);
  LOAD_SYMBOL(hDll, llmModel_Load);
  LOAD_SYMBOL(hDll, llmModel_GetName);
  LOAD_SYMBOL(hDll, llmPrompt_New);
  LOAD_SYMBOL(hDll, llmPrompt_Delete);
  LOAD_SYMBOL(hDll, llmPrompt_AppendText);
  LOAD_SYMBOL(hDll, llmPrompt_AppendControlToken);
  LOAD_SYMBOL(hDll, llmPrompt_AppendAudio);
  LOAD_SYMBOL(hDll, llmCompletion_New);
  LOAD_SYMBOL(hDll, llmCompletion_Delete);
  LOAD_SYMBOL(hDll, llmCompletion_SetPrompt);
  LOAD_SYMBOL(hDll, llmCompletion_SetTopP);
  LOAD_SYMBOL(hDll, llmCompletion_SetTopK);
  LOAD_SYMBOL(hDll, llmCompletion_SetTemperature);
  LOAD_SYMBOL(hDll, llmCompletion_SetConfig);
  LOAD_SYMBOL(hDll, llmCompletion_Next);
  LOAD_SYMBOL(hDll, llmCompletion_GetError);
  LOAD_SYMBOL(hDll, llmCompletion_GetText);
  LOAD_SYMBOL(hDll, llmCompletion_GetToken);

  return LLM_OK;
}

// load the libllm shared library.
llmStatus_t llmDestroyLibrary(LLM_HMODULE handle) {
  p_llmInit = NULL;
  p_llmDestroy = NULL;
  p_llmGetLastErrorMessage = NULL;
  p_llmModel_New = NULL;
  p_llmModel_Delete = NULL;
  p_llmModel_SetFile = NULL;
  p_llmModel_SetDevice = NULL;
  p_llmModel_Load = NULL;
  p_llmModel_GetName = NULL;
  p_llmPrompt_New = NULL;
  p_llmPrompt_Delete = NULL;
  p_llmPrompt_AppendText = NULL;
  p_llmPrompt_AppendControlToken = NULL;
  p_llmPrompt_AppendAudio = NULL;
  p_llmCompletion_New = NULL;
  p_llmCompletion_Delete = NULL;
  p_llmCompletion_SetPrompt = NULL;
  p_llmCompletion_SetTopP = NULL;
  p_llmCompletion_SetTopK = NULL;
  p_llmCompletion_SetTemperature = NULL;
  p_llmCompletion_SetConfig = NULL;
  p_llmCompletion_Next = NULL;
  p_llmCompletion_GetError = NULL;
  p_llmCompletion_GetText = NULL;
  p_llmCompletion_GetToken = NULL;

  // first try to load the dll from same folder as current module.
#if defined(LUT_PLATFORM_APPLE) || defined(LUT_PLATFORM_LINUX)
  int ret = dlclose(handle);
  if (ret != 0) {
    return LLM_ABORTED;
  }
#elif defined(LUT_PLATFORM_WINDOWS)
  BOOL success = FreeLibrary(handle);
  if (FALSE == success) {
    return LLM_ABORTED;
  }
#endif

  return LLM_OK;
}

llmStatus_t llmInit(int32_t apiVersion) {
  return p_llmInit(apiVersion);
}

llmStatus_t llmDestroy() {
  return p_llmDestroy();
}

const char *llmGetLastErrorMessage() {
  return p_llmGetLastErrorMessage();
}

// llmModel_t
llmModel_t *llmModel_New() {
  return p_llmModel_New();
}

llmStatus_t llmModel_Delete(llmModel_t *model) {
  return p_llmModel_Delete(model);
}

llmStatus_t llmModel_SetFile(llmModel_t *model, const char *filename) {
  return p_llmModel_SetFile(model, filename);
}

llmStatus_t llmModel_SetDevice(llmModel_t *model, int32_t device) {
  return p_llmModel_SetDevice(model, device);
}

llmStatus_t llmModel_Load(llmModel_t *model) {
  return p_llmModel_Load(model);
}

const char *llmModel_GetName(llmModel_t *model) {
  return p_llmModel_GetName(model);
}

// llmPrompt_t
llmPrompt_t *llmPrompt_New() {
  return p_llmPrompt_New();
}

llmStatus_t llmPrompt_Delete(llmPrompt_t *prompt) {
  return p_llmPrompt_Delete(prompt);
}

llmStatus_t llmPrompt_AppendText(llmPrompt_t *prompt, const char *text) {
  return p_llmPrompt_AppendText(prompt, text);
}

llmStatus_t llmPrompt_AppendControlToken(llmPrompt_t *prompt, const char *token) {
  return p_llmPrompt_AppendControlToken(prompt, token);
}

llmStatus_t llmPrompt_AppendAudio(
    llmPrompt_t *prompt,
    const llmByte_t *audio,
    int64_t size,
    int32_t format) {
  return p_llmPrompt_AppendAudio(prompt, audio, size, format);
}

// llmCompletion_t
llmCompletion_t *llmCompletion_New(llmModel_t *model) {
  return p_llmCompletion_New(model);
}

llmStatus_t llmCompletion_Delete(llmCompletion_t *comp) {
  return p_llmCompletion_Delete(comp);
}

llmStatus_t llmCompletion_SetPrompt(llmCompletion_t *comp, llmPrompt_t *prompt) {
  return p_llmCompletion_SetPrompt(comp, prompt);
}

llmStatus_t llmCompletion_SetTopP(llmCompletion_t *comp, float topP) {
  return p_llmCompletion_SetTopP(comp, topP);
}

llmStatus_t llmCompletion_SetTopK(llmCompletion_t *comp, int32_t topK) {
  return p_llmCompletion_SetTopK(comp, topK);
}

llmStatus_t llmCompletion_SetTemperature(llmCompletion_t *comp, float temperature) {
  return p_llmCompletion_SetTemperature(comp, temperature);
}

llmStatus_t llmCompletion_SetConfig(llmCompletion_t *comp, const char *key, const char *value) {
  return p_llmCompletion_SetConfig(comp, key, value);
}

llmBool_t llmCompletion_Next(llmCompletion_t *comp) {
  return p_llmCompletion_Next(comp);
}

llmStatus_t llmCompletion_GetError(llmCompletion_t *comp) {
  return p_llmCompletion_GetError(comp);
}

const char *llmCompletion_GetText(llmCompletion_t *comp) {
  return p_llmCompletion_GetText(comp);
}

const char *llmCompletion_GetToken(llmCompletion_t *comp) {
  return p_llmCompletion_GetToken(comp);
}