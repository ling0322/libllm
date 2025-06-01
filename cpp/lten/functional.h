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

#include "lten/tensor.h"
#include "lutil/random.h"
#include "lutil/span.h"

namespace lten {
namespace F {

// retrieve word embeddings using indices. Input is a long tensor with indices and the output is
// the word embeddings for these indices.
// Args:
//   table <float>(V, D): the embedding table. V is vocab size and D is the embedding dimension.
//   indices <long>(N, L): the indices.
// Returns:
//   <float>(N, L, D): the word embedding tensor.
Tensor lookup(Tensor table, Tensor indices);

// apply layer normalization over the last dimension of inputs.
//   y_ij = (x_ij - E[x]) / sqrt(Var[X] + eps)
//   y_ij = y_ij * weight_j + bias_j
// Args:
//   input <float>(..., D): input tensor.
//   weight <float>(D): weight tensor.
//   bias <float>(D): bias tensor.
// Return:
//   <float>(..., D): layer normalized.
Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps);

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

// Apply softmax on the last dimension of input
Tensor softmax(Tensor input);

// return input + other.
Tensor add(Tensor input, Tensor other);

// Applies the Gaussian Error Linear Units function for `input`. Here it use the approximate
// version of GELU:
//   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
// Args:
//   input <float>(..., D): input tensor.
// Returns:
//   <float>(..., D): outpur tensor.
Tensor gelu(Tensor input);

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
/// @param generator random number generator. nullptr for using a default one,
/// @param min minimal value for random number generator.
/// @param max maximum value for random number generator.
/// @return Generated random tensor.
Tensor rand(
    lut::Span<const int> shape,
    DType dtype,
    Device device = Device::getCpu(),
    lut::Random *generator = nullptr,
    float min = -1.0f,
    float max = 1.0f);

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

// Apply rotary position embedding to tensor A.
// Args:
//   A <float>(N, L, nHead, D): the input tensor.
//   roPE <float>(L, 1, D): the tensor of rotary position embedding. [..., ::0] is the values of
//       cos(m*theta), [..., ::1] is the values of sin(m*theta).
// Returns:
//   <float>(N, L, nHead, D): the output tensor.
Tensor applyRotaryPosEmb(Tensor A, Tensor roPE);

// Copy elements from src to dest. Shapes of `src` and `dest` should be the same.
void copy(Tensor src, Tensor dest);

// Compute the scaled dot product attention for given QKV and mask.
// Args:
//   q <float>(N, nHead, L, D): the query.
//   k <float>(N, nHead, S, D): the key.
//   v <float>(N, nHead, S, D): the value.
//   mask <float>(L, S):  A float mask added to the attention score.
// Returns:
//   <float>(N, nHead, L, D): the output tensor.
Tensor attention(Tensor q, Tensor k, Tensor v, Tensor mask = Tensor());

// Applies the Swish-Gated Linear Unit function SwiGLU(a, b) = swish(a) * b.  Where a is the first
// half of input (input[..., :input.shape[-1] / 2]) and b is the second half of input
// (input[..., input.shape[-1] / 2 :]).
// Args:
//   input <float>(..., D): the input (D % 2 == 0).
// Returns:
//   <float>(..., D / 2): the output tensor.
Tensor swiglu(Tensor input);

/// @brief Apply Gaussian error linear unit (GELU) activation to the inputs. it applies
/// element-wise the function GELU(x) = x * Phi(x) where Phi(x) is  the Cumulative Distribution. In
/// the implementation, it did not use the approximate version. Function for Gaussian Distribution.
/// @param inputs: <float>(..., D): the input tensor.
/// @return <float>(..., D): the output tensor.
Tensor gelu(Tensor inputs);

/// @brief fill tensor with value.
/// @param tensor the tensor to fill.
/// @param value the value.
void fill(Tensor tensor, float value);

/// @brief Returns the sum of each row of the input tensor in the given dimension dim.
/// @param tensor <float>(d1, d2, ..., dn) the input tensor.
/// @return <float>(d1, d2, ..., dn-1): the output tensor.
Tensor sum(Tensor tensor, int dim = -1);

/// @brief Returns the maximum value of each row of the input tensor in the given dimension dim.
/// @param tensor <float>(d1, d2, ..., dn) the input tensor.
/// @return <float>(d1, d2, ..., dn-1): the output tensor.
Tensor max(Tensor tensor, int dim = -1);

/// @brief (im2col) Extracts sliding local blocks from the input tensor. To make
/// sure the input and output shape are the same after Conv, it will also pad the input tensor with
/// zero.
/// @param input <float>(N, L, C): the input tensor.
/// @param kernelSize: the kernel size.
/// @param stride: the stride.
/// @return  <float>(N, L / stride, D * kernelSize): the output tensor.
Tensor unfold(Tensor input, int kernelSize, int stride);

/// @brief Extract the log mel spectrogram feature from input wave.
/// @param wave <float>(wave_len, ): the input wave.
/// @return <float>(feature_len, FeatDim=80): the logMelSpectrogram feature.
Tensor logMelSpectrogram(Tensor wave);

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

}  // namespace F
}  // namespace lten
