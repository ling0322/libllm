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

#ifndef LIBLLM_LLM_API_
#define LIBLLM_LLM_API_

#include <dlfcn.h>
#include <linux/limits.h>
#include <stdint.h>
#include <stdio.h>

#define LLM_DEVICE_CPU 0x0000
#define LLM_DEVICE_CUDA 0x0100
#define LLM_DEVICE_AUTO 0x1f00
#define LLM_API_VERSION 20240101
#define LLM_OK 0
#define LLM_ABORTED 1

typedef int32_t llmStatus_t;
typedef struct llmModel_t llmModel_t;
typedef struct llmChunk_t llmChunk_t;
typedef struct llmPrompt_t llmPrompt_t;
typedef struct llmCompletion_t llmCompletion_t;
typedef int32_t llmBool_t;

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
llmPrompt_t *(*p_llmPrompt_New)(llmModel_t *model);
llmStatus_t (*p_llmPrompt_Delete)(llmPrompt_t *prompt);
llmStatus_t (*p_llmPrompt_AppendText)(llmPrompt_t *prompt, const char *text);
llmStatus_t (*p_llmPrompt_AppendControlToken)(llmPrompt_t *prompt, const char *token);

// llmCompletion_t
llmCompletion_t *(*p_llmCompletion_New)(llmModel_t *model);
llmStatus_t (*p_llmCompletion_Delete)(llmCompletion_t *comp);
llmStatus_t (*p_llmCompletion_SetPrompt)(llmCompletion_t *comp, llmPrompt_t *prompt);
llmStatus_t (*p_llmCompletion_SetTopP)(llmCompletion_t *comp, float topP);
llmStatus_t (*p_llmCompletion_SetTopK)(llmCompletion_t *comp, int32_t topK);
llmStatus_t (*p_llmCompletion_SetTemperature)(llmCompletion_t *comp, float temperature);
llmStatus_t (*p_llmCompletion_Start)(llmCompletion_t *comp);
llmBool_t (*p_llmCompletion_IsActive)(llmCompletion_t *comp);
llmStatus_t (*p_llmCompletion_GenerateNextChunk)(llmCompletion_t *comp, llmChunk_t *chunk);

// llmChunk_t
llmChunk_t *(*p_llmChunk_New)();
llmStatus_t (*p_llmChunk_Delete)(llmChunk_t *chunk);
const char *(*p_llmChunk_GetText)(llmChunk_t *chunk);

// load the libllm shared library.
void *llmLoadLibrary(const char *libraryPath) {
  // first try to load the dll from same folder as current module.
  return dlopen(libraryPath, RTLD_NOW);
}

#define LOAD_SYMBOL(hDll, symbol)                                    \
  p_##symbol = dlsym(hDll, #symbol);                                 \
  if (!p_##symbol) {                                                 \
    fprintf(stderr, "llm.go: unable to load symbol: %s\n", #symbol); \
    return LLM_ABORTED;                                              \
  }

llmStatus_t llmLoadSymbols(void *hDll) {
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
  LOAD_SYMBOL(hDll, llmCompletion_New);
  LOAD_SYMBOL(hDll, llmCompletion_Delete);
  LOAD_SYMBOL(hDll, llmCompletion_SetPrompt);
  LOAD_SYMBOL(hDll, llmCompletion_SetTopP);
  LOAD_SYMBOL(hDll, llmCompletion_SetTopK);
  LOAD_SYMBOL(hDll, llmCompletion_SetTemperature);
  LOAD_SYMBOL(hDll, llmCompletion_Start);
  LOAD_SYMBOL(hDll, llmCompletion_IsActive);
  LOAD_SYMBOL(hDll, llmCompletion_GenerateNextChunk);
  LOAD_SYMBOL(hDll, llmChunk_New);
  LOAD_SYMBOL(hDll, llmChunk_Delete);
  LOAD_SYMBOL(hDll, llmChunk_GetText);

  return LLM_OK;
}

// load the libllm shared library.
llmStatus_t llmDestroyLibrary(void *handle) {
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
  p_llmCompletion_New = NULL;
  p_llmCompletion_Delete = NULL;
  p_llmCompletion_SetPrompt = NULL;
  p_llmCompletion_SetTopP = NULL;
  p_llmCompletion_SetTopK = NULL;
  p_llmCompletion_SetTemperature = NULL;
  p_llmCompletion_Start = NULL;
  p_llmCompletion_IsActive = NULL;
  p_llmCompletion_GenerateNextChunk = NULL;
  p_llmChunk_New = NULL;
  p_llmChunk_Delete = NULL;
  p_llmChunk_GetText = NULL;

  // first try to load the dll from same folder as current module.
  int ret = dlclose(handle);
  if (ret != 0) {
    return LLM_ABORTED;
  }

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
llmPrompt_t *llmPrompt_New(llmModel_t *model) {
  return p_llmPrompt_New(model);
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

llmStatus_t llmCompletion_Start(llmCompletion_t *comp) {
  return p_llmCompletion_Start(comp);
}

llmBool_t llmCompletion_IsActive(llmCompletion_t *comp) {
  return p_llmCompletion_IsActive(comp);
}

llmStatus_t llmCompletion_GenerateNextChunk(llmCompletion_t *comp, llmChunk_t *chunk) {
  return p_llmCompletion_GenerateNextChunk(comp, chunk);
}

// llmChunk_t
llmChunk_t *llmChunk_New() {
  return p_llmChunk_New();
}

llmStatus_t llmChunk_Delete(llmChunk_t *chunk) {
  return p_llmChunk_Delete(chunk);
}

const char *llmChunk_GetText(llmChunk_t *chunk) {
  return p_llmChunk_GetText(chunk);
}

#endif  // LIBLLM_LLM_API_
