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

/// Apply NeoX-style rotary embedding to query and key in place. positions is <long>(numTokens),
/// query and key are <float>(numTokens, numHeads, headDim), and rotaryCache is
/// <float>(maxPositions, 2 * headDim) with cosine followed by sine values in each row.
void rotaryEmbedding(Tensor positions, Tensor query, Tensor key, Tensor rotaryCache);

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

/// Sample one label per row from logits using per-row temperature, top-k, and top-p parameters.
/// logits is <float>(rows, vocabSize), temperatures/topPs are <float>(rows), and topKs is
/// <int>(rows). temperature=0 selects greedily; topK<=0 disables top-k filtering.
Tensor sample(Tensor logits, Tensor temperatures, Tensor topKs, Tensor topPs);

// Apply x^2
Tensor square(Tensor input);

/// Element-wise functions of one tensor. Each returns a new tensor of the same shape and type;
/// none of them modify the input.
Tensor neg(Tensor input);
Tensor abs(Tensor input);
Tensor exp(Tensor input);
Tensor sqrt(Tensor input);
/// Reciprocal square root, 1/sqrt(x).
Tensor rsqrt(Tensor input);
Tensor sigmoid(Tensor input);
Tensor tanh(Tensor input);
Tensor relu(Tensor input);
/// The exact Gaussian error linear unit, x * P(X <= x), matching torch.nn.GELU() rather than its
/// tanh approximation.
Tensor gelu(Tensor input);
/// The sigmoid linear unit, x * sigmoid(x). This is the activation swiglu applies to its gate
/// half; here it is available on its own.
Tensor silu(Tensor input);

// return input + other.
Tensor add(Tensor input, Tensor other);

// return input - other.
Tensor sub(Tensor input, Tensor other);

// return input - other.
Tensor div(Tensor input, float other);

/// Element-wise division. `other` is broadcast to the shape of `input` the way mul does.
Tensor div(Tensor input, Tensor other);

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

// Compute the scaled dot product attention of a packed (varlen) batch of queries that reads its
// keys and values from a paged KV cache. Sequence i owns the blocks named by row i of blockTable
// and attends to the first seqlensK[i] tokens they hold. The tokens it already had before this
// call are seqlensK[i] minus its query length, which is where the causal mask starts.
// Args:
//   q <float>(totalQLen, nHead, D): the queries of every sequence, packed back to back.
//   keyCache <float>(nBlock, blockSize, nKvHead, D): the key block pool.
//   valueCache <float>(nBlock, blockSize, nKvHead, D): the value block pool.
//   blockTable <int>(nSeq, maxNumBlock): the blocks each sequence owns, in token order.
//   cuSeqlensQ <int>(nSeq + 1): exclusive prefix sum of the query lengths.
//   seqlensK <int>(nSeq): the number of cached tokens each sequence attends to.
//   maxQLen: the longest query length in the batch.
//   maxKLen: the longest value in seqlensK.
//   causal: mask the future positions, aligned to the bottom right of the score matrix.
// Returns:
//   <float>(totalQLen, nHead, D): the output tensor.
Tensor pagedAttention(
    Tensor q,
    Tensor keyCache,
    Tensor valueCache,
    Tensor blockTable,
    Tensor cuSeqlensQ,
    Tensor seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal);

// Scatter the keys and values a forward pass just produced into a paged KV cache, so a later
// pagedAttention() reads them back.
// Args:
//   k <float>(numTokens, nKvHead, D): the keys, packed like the queries.
//   v <float>(numTokens, nKvHead, D): the values.
//   keyCache <float>(nBlock, blockSize, nKvHead, D): the key block pool, written in place.
//   valueCache <float>(nBlock, blockSize, nKvHead, D): the value block pool, written in place.
//   slotMapping <int>(numTokens): the slot of each token, as blockId * blockSize + offset.
void storeKVCache(Tensor k, Tensor v, Tensor keyCache, Tensor valueCache, Tensor slotMapping);

// Gated DeltaNet linear attention over a packed (varlen) batch, the prefill form: every sequence
// runs from its own incoming recurrent state through all of its tokens at once. Each value head
// carries a (D, D) state S and applies, per token,
//   S_t = exp(g_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t
// which the operator evaluates a chunk at a time rather than a token at a time. q and k carry
// nKHead heads against v's nVHead, a multiple of it, so value head h reads query and key head
// h / (nVHead / nKHead) and grouped keys need no expanded copy. g and beta belong to the value
// head, since the state does.
// Args:
//   q <float>(numTokens, nKHead, D): the queries of every sequence, packed back to back.
//   k <float>(numTokens, nKHead, D): the keys.
//   v <float>(numTokens, nVHead, D): the values.
//   g <float32>(numTokens, nVHead): the log decay of each step, at most 0.
//   beta <float32>(numTokens, nVHead): the delta rule write strength of each step.
//   cuSeqlens <int>(nSeq + 1): exclusive prefix sum of the sequence lengths.
//   stateSlots <int>(nSeq): the slot of the pool each sequence's state occupies. Sequence i reads
//       its incoming state from state[stateSlots[i]] and overwrites that same slot, so a sequence
//       is tied to its state by the mapping rather than by its position in the batch -- the way a
//       paged KV cache reaches its blocks through a block table, at one slot per sequence because
//       a linear attention's whole history is one fixed-size state. A slot therefore outlives the
//       batch: the same sequence may sit anywhere in the next prefill and still find its state.
//       Two sequences may not name the same slot -- each is written by its own blocks and nothing
//       orders them against each other.
//   state <float32>(nSlot, nVHead, D, D): the pool of recurrent states, at least nSeq slots of it,
//       read at each sequence's first token and overwritten in place with the state after its last
//       one. Slots no sequence in this batch maps to are left untouched.
// Returns:
//   <float>(numTokens, nVHead, D): the output tensor.
Tensor gatedDeltaNetPrefill(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor g,
    Tensor beta,
    Tensor cuSeqlens,
    Tensor stateSlots,
    Tensor state);

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

/// Smallest element of dimension `dim`, which the result drops the same way sum() does. Only the
/// last dimension is supported.
Tensor min(Tensor tensor, int dim = -1);

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
