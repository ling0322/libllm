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

#if defined(_WIN32)
#define LLMAPI __declspec(dllexport)
#else
#define LLMAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define LLM_DEVICE_CPU  0x0000
#define LLM_DEVICE_CUDA 0x0100
#define LLM_DEVICE_AUTO 0x1f00
#define LLM_API_VERSION 20240101
#define LLM_OK 0

typedef int32_t llmStatus_t;
typedef int32_t llmModel_t;
typedef int32_t llmChunk_t;
typedef int32_t llmPrompt_t;
typedef int32_t llmCompletion_t;
typedef int32_t llmBool_t;

typedef  struct llmApi_t {
  llmStatus_t (*init)();
  llmStatus_t (*destroy)();
  const char *(*getLastErrorMessage)();

  llmModel_t *(*createModel)();
  llmStatus_t (*destroyModel)();
  llmStatus_t (*setModelConfig)(llmModel_t *model, const char *filename);
  llmStatus_t (*setModelDevice)(llmModel_t *model, int device);
  llmStatus_t (*loadModel)(llmModel_t *model);
  const char *(*getModelName)(llmModel_t *model);

  llmPrompt_t *(*createPrompt)();
  llmStatus_t (*destroyPrompt)(llmPrompt_t *prompt);
  llmStatus_t (*appendText)(llmPrompt_t *prompt, const char *text);
  llmStatus_t (*appendToken)(llmPrompt_t *prompt, const char *token);

  llmCompletion_t *(*createCompletion)();
  llmStatus_t (*destroyCompletion)(llmCompletion_t *compl);
  llmStatus_t (*setCompletionModel)(llmCompletion_t *compl, llmModel_t *model);
  llmStatus_t (*setPrompt)(llmCompletion_t *compl, llmPrompt_t *prompt);
  llmStatus_t (*setTopP)(llmCompletion_t *compl, float topP);
  llmStatus_t (*setTopK)(llmCompletion_t *compl, int topK);
  llmStatus_t (*setTemperature)(llmCompletion_t *compl, float temperature);
  llmStatus_t (*startCompletion)(llmCompletion_t *compl);
  llmStatus_t (*isActive)(llmCompletion_t *compl, float temperature);
  llmStatus_t (*getNextChunk)(llmCompletion_t *compl, llmChunk_t *chunk);

  llmChunk_t *(*createChunk)();
  llmStatus_t (*destroyChunk)(llmChunk_t *chunk);
  const char *(*getChunkText)(llmChunk_t *chunk);
} llmApi_t;

LLMAPI const llmApi_t *llmGetApi(int32_t version);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
