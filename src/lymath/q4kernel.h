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
#include <memory>
#include "lymath/args.h"
#include "lyutil/log.h"


namespace lymath {


struct AxpyQ4NotImplKernel {
  static void applyColumn(const Q4GemvArgs &args, int col, PFp32 y) {
    NOT_IMPL();
  }
};

struct DotQ4FallbackKernel {
  static float apply(int64_t n, PCFp32 x, DataQ4 y, int64_t offsetY);
  static float applyRow(const Q4GemvArgs &args, int row);
};

struct DotQ4Avx2Kernel {
  static float apply(int64_t n, PCFp32 x, DataQ4 y, int64_t offsetY);
  static float applyRow(const Q4GemvArgs &args, int row);
};

struct DequantQ4Avx2Kernel {
  static void apply(int n, DataQ4 x, int64_t offsetX, PFp32 y);
};

struct DequantQ4FallbackKernel {
  static void apply(int n, DataQ4 x, int64_t offsetX, PFp32 y);
};

}  // namespace lymath
