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
#ifdef LIBLLM_EXPORTS
#define LLMAPI __declspec(dllexport)
#else  // LIBLLM_EXPORTS
#define LLMAPI __declspec(dllimport)
#endif  // LIBLLM_EXPORTS
#else   // _WIN32
#ifdef LIBLLM_EXPORTS
#define LLMAPI __attribute__((visibility("default")))
#else  // LIBLLM_EXPORTS
#define LLMAPI
#endif  // LIBLLM_EXPORTS
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define LLM_DEVICE_CPU 0x0000
#define LLM_DEVICE_CUDA 0x0100
#define LLM_DEVICE_AUTO 0x1f00
#define LLM_WAVE_FORMAT_PCM16KHZ16BITMONO 0x0001
#define LLM_API_VERSION 20240101
#define LLM_TRUE 1
#define LLM_FALSE 0
#define LLM_OK 0

typedef int32_t llmStatus_t;
typedef struct llmModel_t llmModel_t;
typedef struct llmChunk_t llmChunk_t;
typedef struct llmPrompt_t llmPrompt_t;
typedef struct llmCompletion_t llmCompletion_t;
typedef struct llmLogitsFilter_t llmLogitsFilter_t;
typedef int32_t llmBool_t;
typedef int8_t llmByte_t;

// global state
LLMAPI llmStatus_t llmInit(int32_t apiVersion);
LLMAPI llmStatus_t llmDestroy();
LLMAPI const char *llmGetLastErrorMessage();

// llmModel_t
LLMAPI llmModel_t *llmModel_New();
LLMAPI llmStatus_t llmModel_Delete(llmModel_t *model);
LLMAPI llmStatus_t llmModel_SetFile(llmModel_t *model, const char *filename);
LLMAPI llmStatus_t llmModel_SetDevice(llmModel_t *model, int32_t device);
LLMAPI llmStatus_t llmModel_Load(llmModel_t *model);
LLMAPI const char *llmModel_GetName(llmModel_t *model);

// llmPrompt_t
LLMAPI llmPrompt_t *llmPrompt_New();
LLMAPI llmStatus_t llmPrompt_Delete(llmPrompt_t *prompt);
LLMAPI
llmStatus_t llmPrompt_AppendAudio(
    llmPrompt_t *prompt,
    const llmByte_t *audio,
    int64_t size,
    int32_t format);

LLMAPI llmStatus_t llmPrompt_AppendText(llmPrompt_t *prompt, const char *text);
LLMAPI llmStatus_t llmPrompt_AppendControlToken(llmPrompt_t *prompt, const char *token);

// llmCompletion_t
LLMAPI llmCompletion_t *llmCompletion_New(llmModel_t *model);
LLMAPI llmStatus_t llmCompletion_Delete(llmCompletion_t *comp);
LLMAPI llmStatus_t llmCompletion_SetPrompt(llmCompletion_t *comp, llmPrompt_t *prompt);
LLMAPI llmStatus_t llmCompletion_SetTopP(llmCompletion_t *comp, float topP);
LLMAPI llmStatus_t llmCompletion_SetTopK(llmCompletion_t *comp, int32_t topK);
LLMAPI
llmStatus_t llmCompletion_SetConfig(llmCompletion_t *comp, const char *key, const char *value);
LLMAPI llmStatus_t llmCompletion_SetTemperature(llmCompletion_t *comp, float temperature);
LLMAPI llmBool_t llmCompletion_Next(llmCompletion_t *comp);
LLMAPI llmStatus_t llmCompletion_GetError(llmCompletion_t *comp);
LLMAPI const char *llmCompletion_GetText(llmCompletion_t *comp);

/// @brief Get the name of last generated token.
/// For a normal token the llmCompletion_GetText() will return its byte piece, for example, "foo ";
/// llmCompletion_GetToken() will return its name in model, for example, "hello_".
/// For a control token, llmCompletion_GetText() will return an empty string ("") and
/// llmCompletion_GetToken() will return its name, for example, "<|endoftext|>".
/// @param comp the llmCompletion_t.
/// @return name of the token.
LLMAPI const char *llmCompletion_GetToken(llmCompletion_t *comp);

// Basic json functions
typedef struct llm_json_t llm_json_t;

LLMAPI llm_json_t *llm_json_parse(const char *json_string);

/// @brief dump the json to string. The returned string pointer is vaild until another
/// llm_json_dump() call.
/// @param json the json to json.
/// @return string of the json.
LLMAPI const char *llm_json_dump(llm_json_t *json);

LLMAPI void llm_json_delete(llm_json_t *json);
LLMAPI const char *llm_json_get_last_error_message();

// ASR

typedef struct llm_asr_recognizer_t llm_asr_recognizer_t;

/// @brief Recognize a media file.
/// @param model the model for ASR.
/// @param options json object of options
///    {
///      "media_file": "<media_file>", // REQUIRED
///    }
/// @return an instance of llm_asr_recognizer_t.
LLMAPI llm_asr_recognizer_t *llm_asr_recognize_media_file(llmModel_t *model, llm_json_t *options);

/// @brief Delete an instance of llm_asr_recognizer_t.
/// @param r the llm_asr_recognizer_t to delete.
LLMAPI void llm_asr_recognizer_delete(llm_asr_recognizer_t *r);

/// @brief Get next recognition event. If no more events, returns a NULL pointer.
/// @param r the llm_asr_recognizer_t.
/// @return the next recognition event in json. Common events:
/// recognition_result:
/// {
///   "event_type": "recognition_result",
///   "content": {
///      "text": "<text>",
///      “begin": <begin_offset_in_ms>,
///      "end": <end_offset_in_ms>,
//       "language": "<language_of_the_text>",
///   }
/// }
/// error:
/// {
///   "event_type": "error",
///   "content": {
///      "reason": "<error_reason>",
///   }
/// }
LLMAPI llm_json_t *llm_asr_recognizer_get_next_event(llm_asr_recognizer_t *r);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
