// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

namespace lten {}  // namespace lten

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct LTensor LTensor;

#define LTEN_ERR_INVALID_ARG 1

#define LTEN_DEVICE_CPU 0x00000000
#define LTEN_DEVICE_CUDA 0x00010000

#define LTEN_DTYPE_FLOAT 1
#define LTEN_DTYPE_INT64 2
#define LTEN_DTYPE_UINT8 3
#define LTEN_DTYPE_FLOAT16 4
#define LTEN_DTYPE_QINT4 5
#define LTEN_DTYPE_INT8 6

#define LTEN_RANGE_NONE -0x1000000000000000

enum LynnOperator {
  LTEN_OP_ADD = 0,
  LTEN_OP_MUL = 1,
  LTEN_OP_ROPE = 2,
  LTEN_OP_SOFTMAX = 3,
  LTEN_OP_GELU = 4,
  LTEN_OP_SWIGLU = 5,
  LTEN_OP_CONTIGUOUS = 6,
  LTEN_OP_SUM = 7,
  LTEN_OP_MAX = 8,
  LTEN_OP_MATMUL = 9,
  LTEN_OP_LOOKUP = 10,
  LTEN_OP_SCALAR_MUL = 11,
  LTEN_OP_LAYER_NORM = 12,
  LTEN_OP_RMS_NORM = 13
};

const char *lten_last_error_message();
int32_t lten_destroy_tensor(LTensor *tensor);
LTensor *lten_new_tensor(int32_t dim, const int64_t *shape, int32_t dtype, int32_t device);

int32_t lten_get_dim(LTensor *tensor, int32_t *dim);
int32_t lten_get_shape(LTensor *tensor, int32_t dim, int64_t *size);
int32_t lten_get_dtype(LTensor *tensor, int32_t *dtype);
int32_t lten_get_device(LTensor *tensor, int32_t *device);
int32_t lten_get_numel(LTensor *tensor, int64_t *numel);
void *lten_get_data_ptr(LTensor *tensor);
LTensor *lten_view(LTensor *tensor, int32_t dim, int64_t *shape);
LTensor *lten_transpose(LTensor *tensor, int32_t dim0, int32_t dim1);
LTensor *lten_expand(LTensor *tensor, int32_t dim, int64_t *shape);
LTensor *lten_slice(LTensor *tensor, int32_t dim, int64_t begin, int64_t end);
LTensor *lten_clone(LTensor *tensor);
LTensor *lten_index(LTensor *tensor, int64_t index);

LTensor *lten_to_device(LTensor *tensor, int32_t device);
LTensor *lten_to_dtype(LTensor *tensor, int32_t dtype);

int32_t lten_copy(LTensor *dest, LTensor *src);
int32_t lten_fill_float(LTensor *tensor, float value);
int32_t lten_print(LTensor *tensor);

LTensor *lten_apply_operator(
    LTensor *targ0,
    LTensor *targ1,
    LTensor *targ2,
    LTensor *targ3,
    int64_t iarg0,
    int64_t iarg1,
    float farg0,
    float farg1,
    int32_t op);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
