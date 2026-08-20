// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include "lutil/random.h"
#include "lutil/span.h"
#include "flint/device.h"
#include "flint/dtype.h"

namespace fl {
class Tensor;

namespace F {

Tensor arange(LongType begin, LongType end, LongType step = 1, Device device = Device::getCpu());

// retrieve word embeddings using indices. Input is a long tensor with indices and the output is
// the word embeddings for these indices.
// Args:
//   table <float>(V, D): the embedding table. V is vocab size and D is the embedding dimension.
//   indices <long>(N, L) or <long>(L): the indices.
// Returns:
//   <float>(N, L, D) or <float>(L, D): the word embedding tensor, one dimension more than
//   `indices`.
Tensor lookup(Tensor table, Tensor indices);

// apply layer normalization over the last dimension of inputs.
// apply root mean square layer normalization over the last dimension of inputs.
// Args:
//   input <float>(..., D): input tensor.
//   weight <float>(D): weight tensor.
//   bias <float>(D): bias tensor.
// Return:
//   <float>(..., D): RMS normalized.
Tensor rmsNorm(Tensor input, Tensor weight, float eps);

// matrix multiplication of tensor A and B. It will dispatch the operator to different routines
// according to the input shape of A and B.
// Args:
//   A <float>(...): tensor A;
//   B <float>(...): tensor B;
// Return:
//   <float>(<batch-dims>, M): matrix multiplication result of A and B.
Tensor matmul(Tensor A, Tensor B);

// Element wise multiply input and other.
Tensor mul(Tensor input, float other);
Tensor mul(Tensor input, Tensor other);

// apply input % other
Tensor mod(Tensor input, LongType other);

// Apply softmax on the last dimension of input
Tensor softmax(Tensor input);

// Sample one label per row from a probability distribution using top-k and nucleus filtering.
Tensor sample(Tensor distribution, int topK, float topP);

// Apply x^2
Tensor square(Tensor input);

// return input + other.
Tensor add(Tensor input, Tensor other);

// return input - other.
Tensor sub(Tensor input, Tensor other);

// return input - other.
Tensor div(Tensor input, float other);

// create a tensor with specified shape and dtype. Data in this tensor is uninitialize.
// Args:
//   shape: shape of the new tensor.
//   dtype: data type of the new tensor.
// Returns:
//   the tensor with specified shape and dtype.
Tensor tensor(lut::Span<const int> shape, DType dtype, Device device = Device::getCpu());

/// @brief Generate a tensor filled with uniform random numbers in range [min, max)
/// @param shape shape of the tensor to generated.
/// @param dtype data type of the tensor.
/// @param device device of the tensor.
/// @return Generated random tensor.
Tensor rand(lut::Span<const int> shape, DType dtype, Device device = Device::getCpu());

/// @brief Returns a tensor filled with random numbers from a normal distribution with mean 0 and
/// variance 1
Tensor randn(lut::Span<const int> shape, Device device = Device::getCpu());

// returns a uninitialized tensor with the same shape and dtype as input
Tensor tensorLike(Tensor input);

// Returns a tensor filled with 0
Tensor zeros(lut::Span<const int> shape, DType dtype, Device device = Device::getCpu());

// Return a contiguous in memory tensor containing the same data as input
Tensor contiguous(Tensor input);

// return true if two tensors are element-wise equal within a tolerance
// (rtol=1e-05, atol=1e-08)
bool allClose(Tensor A, Tensor B, float rtol = 1e-3, float atol = 1e-5);

// Print the tensor to stdout,
void print(Tensor tensor);

// Returns a tensor of causal mask. For the position not allowed to attend it would be +inf, for
// the position allowed leave 0.0f.
// Args:
//   max_len: max length of the sequence.
// Returns:
//   <float>(max_len, max_len): the causal mask.
Tensor causalMask(int max_len, Device device = Device::getCpu());

// Concat two tensors in the given dimension. Besides the dimension to concat, the two tensors
// must have the same shape.
// Args:
//   A: the first tensor.
//   B: the second tensor.
//   dim (int): the dimension to concat alone.
// Returns:
//   C: concatenated tensor.
Tensor cat(Tensor A, Tensor B, int dim);

// Copy elements from src to dest. Shapes of `src` and `dest` should be the same.
void copy(Tensor src, Tensor dest);

// Compute the scaled dot product attention for given QKV. k and v may carry fewer heads than q,
// grouped-query and multi-query attention need no expanded k and v.
// Args:
//   q <float>(N, nHead, L, D): the query.
//   k <float>(N, nKvHead, S, D): the key.
//   v <float>(N, nKvHead, S, D): the value.
//   causal: mask the future positions, aligned to the bottom right of the score matrix.
// Returns:
//   <float>(N, nHead, L, D): the output tensor.
Tensor attention(Tensor q, Tensor k, Tensor v, bool causal);

// Applies the Swish-Gated Linear Unit function SwiGLU(a, b) = swish(a) * b.  Where a is the first
// half of input (input[..., :input.shape[-1] / 2]) and b is the second half of input
// (input[..., input.shape[-1] / 2 :]).
// Args:
//   input <float>(..., D): the input (D % 2 == 0).
// Returns:
//   <float>(..., D / 2): the output tensor.
Tensor swiglu(Tensor input);

/// @brief fill tensor with value.
/// @param tensor the tensor to fill.
/// @param value the value.
void fill(Tensor tensor, float value);

/// @brief Returns the sum of each row of the input tensor in the given dimension dim.
/// @param tensor <float>(d1, d2, ..., dn) the input tensor.
/// @param dim <int>: the dimension to reduce. None for all dimensions.
/// @return <float>(d1, d2, ..., dn-1): the output tensor.
Tensor sum(Tensor tensor, int dim = -1);

/// @brief Returns the maximum value of each row of the input tensor in the given dimension dim.
/// @param tensor <float>(d1, d2, ..., dn) the input tensor.
/// @return <float>(d1, d2, ..., dn-1): the output tensor.
Tensor max(Tensor tensor, int dim = -1);

/// @brief Apply repetition penalty to the logits tensor according to the history tensor.
/// @param logits <float>(N, vocab_size): the logits tensor to apply repetition penalty.
/// @param history <long>(N, hsitory_len): the token history.
/// @param weight weight of repetition penalty.
void repetitionPenalty(Tensor logits, Tensor history, float weight);

/// @brief Copy the tensor to target device. If `castFloat` is true and the tensor type is float,
//         it will cast the data type to default float type of that device.
/// @param tensor the source tensor.
/// @param device target device.
/// @param castFloat if cast the float type.
/// @return the tensor in device.
Tensor to(Device device, Tensor tensor);

/// @brief Cast tensor to another data type.
/// @param tensor Source tensor.
/// @param dtype Target data type.
/// @return tensor with data type `dtype`.
Tensor cast(Tensor tensor, DType dtype);

/// @brief Get default float type of operators for specific device.
/// @param device The device to query.
/// @return float type as DType.
DType getDefaultFloatType(Device device);

/// @brief Get element from scalar tensor (1D tensor with only 1 element)
float elem(Tensor tensor);

/// tensor == other (no reduce)
Tensor eq(Tensor tensor, Tensor other);

/// Returns true if all elements in tensor are true
bool all(Tensor tensor);

void manualSeed(Device device, uint64_t seed);

}  // namespace F
}  // namespace fl
