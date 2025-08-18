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

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <libunwind.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <mutex>

#include "lutil/platform.h"

namespace lut {

std::once_flag gPrintStackOnExitFlag;

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

void printStackTrace() {
  puts("Stack trace:");

  unw_cursor_t cursor;
  unw_context_t uc;
  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);

  int frame = 0;
  while (unw_step(&cursor) > 0) {
    unw_word_t ip = 0, sp = 0, off = 0;
    char sym[512];

    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    unw_get_reg(&cursor, UNW_REG_SP, &sp);

    sym[0] = '\0';
    if (unw_get_proc_name(&cursor, sym, sizeof(sym), &off) != 0) {
      snprintf(sym, sizeof(sym), "???");
      off = 0;
    }

    Dl_info dli = {0};
    const char *obj = "";
    if (dladdr((void *)(uintptr_t)ip, &dli) && dli.dli_fname) obj = dli.dli_fname;

#ifdef __cplusplus
    int status = 0;
    char *dem = abi::__cxa_demangle(sym, 0, 0, &status);
    const char *name = (status == 0 && dem) ? dem : sym;
#else
    const char *name = sym;
#endif

    fprintf(
        stderr,
        "#%-2d  %p  %s + 0x%lx  (%s)\n",
        frame++,
        (void *)(uintptr_t)ip,
        name,
        (unsigned long)off,
        obj);
#ifdef __cplusplus
    if (status == 0 && name != sym) free((void *)name);
#endif
  }
}

}  // namespace lut
