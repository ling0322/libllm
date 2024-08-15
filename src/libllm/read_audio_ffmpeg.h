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

LLMAPI char *llm_ffmpeg_plugin_get_err();
LLMAPI int32_t llm_ffmpeg_plugin_read_16khz_mono_pcm_from_media_file(
    const char *filename,
    char **output_buffer,
    int32_t *output_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
