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

#include "lyutil/platform.h"

#include <stdlib.h>

namespace lut {

void initCpuInfo() {
#if !defined(__clang__) || __clang_major__ >= 6
  __builtin_cpu_init();
#endif
}

bool isAvx512Available() {
  initCpuInfo();
  return __builtin_cpu_supports("avx512f") != 0;
}

bool isAvx2Available() {
  initCpuInfo();
  return __builtin_cpu_supports("avx2") != 0;
}


void *alloc32ByteAlignedMem(int64_t size) {
  if (size % 32 != 0) {
    size += (32 - size % 32);
  }
  return aligned_alloc(32, size);
}

void free32ByteAlignedMem(void *ptr) {
  free(ptr);
}

const char *getPathDelim() {
  return "/";
}

} // namespace lut
