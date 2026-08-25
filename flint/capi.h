// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

/// C interface to the flint tensor, meant for language bindings. Every function is thread-safe in
/// the sense that it has no shared mutable state of its own, but a single tensor handle must not
/// be used from two threads at once.

#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef LIBLLM_EXPORTS
#define FLAPI __declspec(dllexport)
#else  // LIBLLM_EXPORTS
#define FLAPI __declspec(dllimport)
#endif  // LIBLLM_EXPORTS
#else   // _WIN32
#ifdef LIBLLM_EXPORTS
#define FLAPI __attribute__((visibility("default")))
#else  // LIBLLM_EXPORTS
#define FLAPI
#endif  // LIBLLM_EXPORTS
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define FL_OK 0
#define FL_ERROR_INVALID_ARG 0x0100
#define FL_ERROR_ABORTED 0x0102

/// Open end of a slice range, standing for "from the start" or "to the end".
#define FL_NONE (-2147483647 - 1)

/// A tensor. Holds a shape and a reference to storage that other tensors may share.
typedef struct fl_tensor_impl_t *fl_tensor_t;

typedef enum fl_dtype_t {
  FL_DTYPE_UNKNOWN = 0,
  FL_DTYPE_FLOAT = 1,
  FL_DTYPE_LONG = 2,
  FL_DTYPE_UINT8 = 3,
  FL_DTYPE_FLOAT16 = 4,
  FL_DTYPE_INT8 = 6,
  FL_DTYPE_FP4E2M0X2 = 7,
  FL_DTYPE_BOOL = 8,
  FL_DTYPE_INT32 = 9,
} fl_dtype_t;

typedef enum fl_device_type_t {
  FL_DEVICE_CPU = 0,
  FL_DEVICE_CUDA = 1,
  FL_DEVICE_UNKNOWN = 3,
} fl_device_type_t;

/// Select the operator backends for the machine. Call once before anything else here; later calls
/// do nothing. Failures are reported through fl_get_last_error_code().
FLAPI void fl_init(void);

/// Whether this build has operators for `device` and the machine can run them. Call fl_init()
/// first; a device that is not available is reported as zero rather than as an error.
FLAPI int32_t fl_is_device_available(fl_device_type_t device, int32_t *out);

/// Error code of the last call made on this thread, or zero if it succeeded.
FLAPI int32_t fl_get_last_error_code(void);

/// Message of the last call made on this thread. Owned by the library, valid until the next call
/// on the same thread, and never NULL.
FLAPI const char *fl_get_last_error_message(void);

/// Create a tensor filled with zeros.
/// @param shape one entry per dimension; may be NULL only when `ndim` is zero.
/// @param out receives the new tensor, which the caller destroys with fl_tensor_destroy().
FLAPI int32_t fl_tensor_zeros(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_device_type_t device,
    fl_tensor_t *out);

/// Create a tensor without writing anything into it, for storage that is about to be overwritten
/// in full. Reading it before writing it gives whatever the allocator handed back.
FLAPI int32_t fl_tensor_empty(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_device_type_t device,
    fl_tensor_t *out);

/// Create a tensor on the CPU holding a copy of `data`.
/// @param data_size size of `data` in bytes; must match the shape and dtype exactly.
FLAPI int32_t fl_tensor_from_data(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    const void *data,
    int64_t data_size,
    fl_tensor_t *out);

/// Create another handle on the same tensor. The storage is shared rather than copied, exactly as
/// it is between a tensor and its views.
FLAPI int32_t fl_tensor_clone(fl_tensor_t tensor, fl_tensor_t *out);

/// Release a handle. Storage goes away once the last handle referring to it is destroyed. Passing
/// NULL has no effect.
FLAPI void fl_tensor_destroy(fl_tensor_t tensor);

/// Number of dimensions.
FLAPI int32_t fl_tensor_get_dim(fl_tensor_t tensor, int32_t *out);

/// Size of dimension `dim`, which may be negative to count from the back.
FLAPI int32_t fl_tensor_get_shape(fl_tensor_t tensor, int32_t dim, int32_t *out);

/// Stride of dimension `dim`, in elements, which may be negative to count from the back.
FLAPI int32_t fl_tensor_get_stride(fl_tensor_t tensor, int32_t dim, int32_t *out);

/// Total number of elements.
FLAPI int32_t fl_tensor_get_numel(fl_tensor_t tensor, int64_t *out);

FLAPI int32_t fl_tensor_get_dtype(fl_tensor_t tensor, fl_dtype_t *out);
FLAPI int32_t fl_tensor_get_device(fl_tensor_t tensor, fl_device_type_t *out);

/// Whether the elements are laid out without gaps, which is what lets a view reinterpret them.
FLAPI int32_t fl_tensor_is_contiguous(fl_tensor_t tensor, int32_t *out);

/// Reinterpret the elements under a new shape, sharing the storage. Requires a contiguous tensor.
FLAPI int32_t fl_tensor_view(
    fl_tensor_t tensor,
    const int32_t *shape,
    int32_t ndim,
    fl_tensor_t *out);

/// Exchange two dimensions, sharing the storage. The result is usually not contiguous.
FLAPI int32_t fl_tensor_transpose(
    fl_tensor_t tensor,
    int32_t dim0,
    int32_t dim1,
    fl_tensor_t *out);

/// Take the half-open range [begin, end) of dimension `dim`, sharing the storage. Either bound may
/// be negative to count from the back, or FL_NONE to leave that end where it is.
FLAPI int32_t fl_tensor_slice(
    fl_tensor_t tensor,
    int32_t dim,
    int32_t begin,
    int32_t end,
    fl_tensor_t *out);

/// Take one entry of the first dimension, dropping that dimension. `index` may be negative.
FLAPI int32_t fl_tensor_subtensor(fl_tensor_t tensor, int32_t index, fl_tensor_t *out);

/// Add a dimension of size one at `dim`.
FLAPI int32_t fl_tensor_unsqueeze(fl_tensor_t tensor, int32_t dim, fl_tensor_t *out);

/// Remove the dimension at `dim`, which must have size one.
FLAPI int32_t fl_tensor_squeeze(fl_tensor_t tensor, int32_t dim, fl_tensor_t *out);

/// Return a contiguous tensor with the same elements, copying only if it is not already one.
FLAPI int32_t fl_tensor_contiguous(fl_tensor_t tensor, fl_tensor_t *out);

/// Copy the tensor to another device.
FLAPI int32_t fl_tensor_to_device(
    fl_tensor_t tensor,
    fl_device_type_t device,
    fl_tensor_t *out);

/// Convert the elements to another data type.
FLAPI int32_t fl_tensor_cast(fl_tensor_t tensor, fl_dtype_t dtype, fl_tensor_t *out);

/// Number of bytes fl_tensor_copy_to_host() will write for this tensor.
FLAPI int32_t fl_tensor_get_nbytes(fl_tensor_t tensor, int64_t *out);

/// Copy the elements into `buffer` in row-major order, moving them from the device and making them
/// contiguous first if needed. `buffer_size` must be at least fl_tensor_get_nbytes().
FLAPI int32_t fl_tensor_copy_to_host(fl_tensor_t tensor, void *buffer, int64_t buffer_size);

/// The operations from `flint/functional.h`. Every one of them writes a freshly created tensor to
/// `out`, or works in place on a tensor the caller already owns.
///
/// They report a null handle or a rejected dtype or device through the error code, as the calls
/// above do, but not everything comes back that way: the operators check the shapes and dtypes
/// they were handed with the library's own fatal check, which ends the process instead of raising
/// anything this layer could catch, and so does asking for an operation the device has no kernel
/// for. Callers are expected to have got the shapes right before they get here.

/// Create a 1-D tensor holding the values of [begin, end) taken `step` at a time.
FLAPI int32_t fl_arange(
    int64_t begin,
    int64_t end,
    int64_t step,
    fl_device_type_t device,
    fl_tensor_t *out);

/// Create a tensor filled with uniform random numbers in [0, 1).
FLAPI int32_t fl_rand(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_device_type_t device,
    fl_tensor_t *out);

/// Create a float tensor filled from a normal distribution with mean 0 and variance 1.
FLAPI int32_t fl_randn(
    const int32_t *shape,
    int32_t ndim,
    fl_device_type_t device,
    fl_tensor_t *out);

/// Seed the random number generator of `device`, which fl_rand() and fl_randn() draw from.
FLAPI int32_t fl_manual_seed(fl_device_type_t device, uint64_t seed);

/// Rows of `table` <float>(V, D) named by `indices` <long>(...), which gains a trailing dimension
/// of D.
FLAPI int32_t fl_lookup(fl_tensor_t table, fl_tensor_t indices, fl_tensor_t *out);

/// Apply NeoX-style rotary embedding to `query` and `key` in place. `positions` is
/// <long>(numTokens), the two others are <float>(numTokens, numHeads, headDim), and
/// `rotary_cache` is <float>(maxPositions, 2 * headDim), each row a cosine half then a sine half.
FLAPI int32_t fl_rotary_embedding(
    fl_tensor_t positions,
    fl_tensor_t query,
    fl_tensor_t key,
    fl_tensor_t rotary_cache);

/// Root mean square layer normalization over the last dimension of `input`, scaled by `weight`
/// <float>(D).
FLAPI int32_t fl_rms_norm(fl_tensor_t input, fl_tensor_t weight, float eps, fl_tensor_t *out);

/// Matrix multiplication, batched over the leading dimensions.
FLAPI int32_t fl_matmul(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out);

/// Element-wise `a` * `b`, broadcasting `b` over the leading dimensions of `a`.
FLAPI int32_t fl_mul(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out);

/// Element-wise `a` + `b`, broadcasting `b` over the leading dimensions of `a`.
FLAPI int32_t fl_add(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out);

/// Element-wise `a` - `b`, broadcasting `b` over the leading dimensions of `a`.
FLAPI int32_t fl_sub(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out);

/// Element-wise `a` == `b`, giving a <bool> tensor of the same shape.
FLAPI int32_t fl_eq(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out);

/// Element-wise `input` * `other`.
FLAPI int32_t fl_mul_scalar(fl_tensor_t input, float other, fl_tensor_t *out);

/// Element-wise `input` / `other`.
FLAPI int32_t fl_div_scalar(fl_tensor_t input, float other, fl_tensor_t *out);

/// Element-wise `input` % `other`, for a <long> tensor.
FLAPI int32_t fl_mod_scalar(fl_tensor_t input, int64_t other, fl_tensor_t *out);

/// Element-wise `input` squared.
FLAPI int32_t fl_square(fl_tensor_t input, fl_tensor_t *out);

/// Softmax over the last dimension.
FLAPI int32_t fl_softmax(fl_tensor_t input, fl_tensor_t *out);

/// Swish-gated linear unit over the last dimension of `input` <float>(..., D), which must be even
/// and which the result halves.
FLAPI int32_t fl_swiglu(fl_tensor_t input, fl_tensor_t *out);

/// Sum over dimension `dim`, which may be negative to count from the back and which the result
/// drops.
FLAPI int32_t fl_sum(fl_tensor_t input, int32_t dim, fl_tensor_t *out);

/// Largest element of dimension `dim`, which the result drops the same way fl_sum() does.
FLAPI int32_t fl_max(fl_tensor_t input, int32_t dim, fl_tensor_t *out);

/// Concatenate `a` and `b` along `dim`. They must agree on every other dimension.
FLAPI int32_t fl_cat(fl_tensor_t a, fl_tensor_t b, int32_t dim, fl_tensor_t *out);

/// A <float>(max_len, max_len) mask holding -inf where a position may not attend and 0 elsewhere.
FLAPI int32_t fl_causal_mask(int32_t max_len, fl_device_type_t device, fl_tensor_t *out);

/// Scaled dot product attention. `q` is <float>(N, nHead, L, D) and `k` and `v` are
/// <float>(N, nKvHead, S, D), where nKvHead may divide nHead for grouped-query attention.
/// `causal` masks the future positions, aligned to the bottom right of the score matrix.
FLAPI int32_t fl_attention(
    fl_tensor_t q,
    fl_tensor_t k,
    fl_tensor_t v,
    int32_t causal,
    fl_tensor_t *out);

/// Scaled dot product attention over a packed batch of queries that reads its keys and values
/// from a paged KV cache. Sequence i owns the blocks named by row i of `block_table` and attends
/// to the first `seqlens_k`[i] tokens they hold.
/// @param q <float>(totalQLen, nHead, D): the queries of every sequence, packed back to back.
/// @param key_cache <float>(nBlock, blockSize, nKvHead, D): the key block pool.
/// @param value_cache <float>(nBlock, blockSize, nKvHead, D): the value block pool.
/// @param block_table <int>(nSeq, maxNumBlock): the blocks each sequence owns, in token order.
/// @param cu_seqlens_q <int>(nSeq + 1): exclusive prefix sum of the query lengths.
/// @param seqlens_k <int>(nSeq): the number of cached tokens each sequence attends to.
FLAPI int32_t fl_paged_attention(
    fl_tensor_t q,
    fl_tensor_t key_cache,
    fl_tensor_t value_cache,
    fl_tensor_t block_table,
    fl_tensor_t cu_seqlens_q,
    fl_tensor_t seqlens_k,
    int32_t max_q_len,
    int32_t max_k_len,
    int32_t causal,
    fl_tensor_t *out);

/// Scatter the keys and values of a forward pass into a paged KV cache, so that a later
/// fl_paged_attention() reads them back. `key_cache` and `value_cache` are written in place, and
/// `slot_mapping` is <int>(numTokens) holding blockId * blockSize + offset per token.
FLAPI int32_t fl_store_kv_cache(
    fl_tensor_t k,
    fl_tensor_t v,
    fl_tensor_t key_cache,
    fl_tensor_t value_cache,
    fl_tensor_t slot_mapping);

/// Sample one label per row of `logits` <float>(rows, vocabSize) with per-row parameters:
/// `temperatures` and `top_ps` are <float>(rows) and `top_ks` is <int>(rows). A temperature of 0
/// selects greedily and a top-k of 0 or less keeps every label.
FLAPI int32_t fl_sample_with_params(
    fl_tensor_t logits,
    fl_tensor_t temperatures,
    fl_tensor_t top_ks,
    fl_tensor_t top_ps,
    fl_tensor_t *out);

/// Divide the logits of the tokens in `history` <long>(N, historyLen) by `weight`, penalizing the
/// ones already generated. `logits` <float>(N, vocabSize) is written in place.
FLAPI int32_t fl_repetition_penalty(fl_tensor_t logits, fl_tensor_t history, float weight);

/// Copy the elements of `src` into `dest`, which must have the same shape.
FLAPI int32_t fl_copy(fl_tensor_t src, fl_tensor_t dest);

/// Fill every element of `tensor` with `value`, in place.
FLAPI int32_t fl_fill(fl_tensor_t tensor, float value);

/// Whether every pair of elements is within `rtol` relative and `atol` absolute tolerance.
FLAPI int32_t fl_all_close(
    fl_tensor_t a,
    fl_tensor_t b,
    float rtol,
    float atol,
    int32_t *out);

/// Whether every element of a <bool> tensor is true.
FLAPI int32_t fl_all(fl_tensor_t tensor, int32_t *out);

/// The single element of a one-element tensor, as a float.
FLAPI int32_t fl_elem(fl_tensor_t tensor, float *out);

/// The float type the operators of `device` work in by default.
FLAPI int32_t fl_get_default_float_type(fl_device_type_t device, fl_dtype_t *out);

/// Print the tensor to stdout.
FLAPI int32_t fl_print(fl_tensor_t tensor);

/// The memory usage of one device. A device that does not report its usage, which is what the CPU
/// backend does, reports zero for every field.
typedef struct fl_memory_snapshot_t {
  int64_t total;
  int64_t free;
  int64_t allocated;
  int64_t peak_allocated;
} fl_memory_snapshot_t;

/// Measure the memory usage of `device`.
FLAPI int32_t fl_memory_capture(fl_device_type_t device, fl_memory_snapshot_t *out);

/// Set the peak allocated bytes of `device` back to zero, so that the next measurement covers only
/// what happens from here.
FLAPI int32_t fl_memory_reset_peak_stats(fl_device_type_t device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
