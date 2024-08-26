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

#include "plugin.h"

#include <stdio.h>
#include <stdlib.h>

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

char *(*p_llm_ffmpeg_get_err)() = NULL;
void *(*p_llm_ffmpeg_audio_open)(const char *filename);
void (*p_llm_ffmpeg_audio_close)(void *reader);
int32_t (*p_llm_ffmpeg_audio_read)(void *reader, char *buf, int32_t buf_size);

// load the libllm shared library.
void *llm_ffmpeg_plugin_load_library(const char *libraryPath) {
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

#define LOAD_SYMBOL(hDll, symbol)                                           \
  p_##symbol = GET_PROC_ADDR(hDll, #symbol);                                \
  if (!p_##symbol) {                                                        \
    fprintf(stderr, "ffmpeg_plugin: unable to load symbol: %s\n", #symbol); \
    return -1;                                                              \
  }

int llm_ffmpeg_plugin_load_symbols(void *pDll) {
  LLM_HMODULE hDll = (LLM_HMODULE)pDll;

  LOAD_SYMBOL(hDll, llm_ffmpeg_get_err);
  LOAD_SYMBOL(hDll, llm_ffmpeg_audio_open);
  LOAD_SYMBOL(hDll, llm_ffmpeg_audio_close);
  LOAD_SYMBOL(hDll, llm_ffmpeg_audio_read);

  return 0;
}

// load the libllm shared library.
void llm_ffmpeg_plugin_destroy_library(void *handle) {
  p_llm_ffmpeg_get_err = NULL;
  p_llm_ffmpeg_audio_open = NULL;
  p_llm_ffmpeg_audio_close = NULL;
  p_llm_ffmpeg_audio_read = NULL;

  // first try to load the dll from same folder as current module.
#if defined(LUT_PLATFORM_APPLE) || defined(LUT_PLATFORM_LINUX)
  int ret = dlclose(handle);
  if (ret != 0) {
    fprintf(stderr, "ffmpeg_plugin: unable to close dl\n");
  }

#elif defined(LUT_PLATFORM_WINDOWS)
  BOOL success = FreeLibrary((LLM_HMODULE)handle);
  if (FALSE == success) {
    fprintf(stderr, "ffmpeg_plugin: unable to close dl\n");
  }
#endif
}

void *llm_ffmpeg_audio_open(const char *filename) {
  return p_llm_ffmpeg_audio_open(filename);
}

void llm_ffmpeg_audio_close(void *reader) {
  return p_llm_ffmpeg_audio_close(reader);
}

int32_t llm_ffmpeg_audio_read(void *reader, char *buf, int32_t buf_size) {
  return p_llm_ffmpeg_audio_read(reader, buf, buf_size);
}

const char *llm_ffmpeg_get_err() {
  return p_llm_ffmpeg_get_err();
}
