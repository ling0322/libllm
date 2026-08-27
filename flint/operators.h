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

#include "lutil/random.h"
#include "lutil/thread_pool.h"
#include "flint/device.h"
#include "flint/memory.h"
#include "flint/tensor.h"

namespace fl {

// base functional interface to apply operators for Tensor
class Operators {
 public:
  virtual ~Operators() = default;

  virtual Tensor arangeLong(LongType begin, LongType end, LongType step);
  virtual Tensor lookup(Tensor table, Tensor indices);
  virtual void rotaryEmbedding(Tensor positions, Tensor query, Tensor key, Tensor rotaryCache);
  virtual Tensor rmsNorm(Tensor input, Tensor weight, float eps);
  virtual Tensor matmul(Tensor A, Tensor B);
  virtual Tensor matmulNarrowPrecision(Tensor A, Tensor sfA, Tensor B, Tensor sfB);

  /// Solve the lower triangular systems L X = B. l is <float>(..., N, N) and b is
  /// <float>(..., N, M) with the same batch dimensions.
  virtual Tensor mul(Tensor input, float other);
  virtual Tensor div(Tensor input, float other);
  virtual Tensor mod(Tensor input, LongType other);
  virtual Tensor mul(Tensor input, Tensor other);
  virtual Tensor softmax(Tensor input);

  /// Scaled dot product attention. q is [batch, numHeads, queryLength, headDim], k and v are
  /// [batch, numKeyValueHeads, keyValueLength, headDim]. numHeads is a multiple of
  /// numKeyValueHeads, so grouped-query and multi-query attention need no expanded k and v.
  virtual Tensor attention(Tensor q, Tensor k, Tensor v, bool causal);

    /// Scaled dot product attention of a packed (varlen) batch of queries over a paged KV cache.
    virtual Tensor pagedAttention(
      Tensor q,
      Tensor keyCache,
      Tensor valueCache,
      Tensor blockTable,
      Tensor cuSeqlensQ,
      Tensor seqlensK,
      int maxQLen,
      int maxKLen,
      bool causal);

    /// Scatter packed keys and values into their paged KV-cache slots.
    virtual void storeKVCache(
      Tensor k,
      Tensor v,
      Tensor keyCache,
      Tensor valueCache,
      Tensor slotMapping);

    /// Gated DeltaNet linear attention over a packed (varlen) batch, prefill form.
    virtual Tensor gatedDeltaNetPrefill(
      Tensor q,
      Tensor k,
      Tensor v,
      Tensor g,
      Tensor beta,
      Tensor cuSeqlens,
      Tensor stateSlots,
      Tensor state);

    virtual Tensor sample(Tensor logits, Tensor temperatures, Tensor topKs, Tensor topPs);
  virtual Tensor add(Tensor input, Tensor other);
  virtual Tensor sub(Tensor input, Tensor other);
  virtual Tensor subFloat(Tensor input, float other);
  virtual Tensor sum(Tensor input, int dim);
  virtual Tensor max(Tensor input);
  virtual Tensor eq(Tensor input, Tensor other);
  virtual Tensor square(Tensor input);
  virtual Tensor min(Tensor input);
  virtual Tensor divTensor(Tensor input, Tensor other);
  virtual Tensor neg(Tensor input);
  virtual Tensor abs(Tensor input);
  virtual Tensor exp(Tensor input);
  virtual Tensor sqrt(Tensor input);
  virtual Tensor rsqrt(Tensor input);
  virtual Tensor sigmoid(Tensor input);
  virtual Tensor tanh(Tensor input);
  virtual Tensor relu(Tensor input);
  virtual Tensor gelu(Tensor input);
  virtual Tensor silu(Tensor input);
  virtual void fill(Tensor input, float value);
  virtual Tensor tensor(lut::Span<const int> shape, DType dtype);
  virtual Tensor tensorLike(Tensor input);
  virtual Tensor zeros(lut::Span<const int> shape, DType dtype);
  virtual bool all(Tensor A);
  virtual bool allClose(Tensor A, Tensor B, float rtol, float atol);
  virtual void print(Tensor tensor);
  virtual Tensor causalMask(int max_len);
  virtual void copy(Tensor src, Tensor dest);
  virtual Tensor swiglu(Tensor A);
  virtual Tensor to(Device device, Tensor tensor);
  virtual float elem(Tensor tensor);
  virtual bool elemBool(Tensor tensor);
  virtual void repetitionPenalty(Tensor logits, Tensor history, float weight);
  virtual Tensor cast(Tensor tensor, DType dtype);
  virtual Tensor rand(lut::Span<const int> shape, DType dtype);
  virtual Tensor randNormal(lut::Span<const int> shape);
  virtual void manualSeed(uint64_t seed);

  virtual MemorySnapshot captureMemorySnapshot();
  virtual void resetPeakMemoryStats();

  virtual DType getDefaultFloatType();
};

Operators *getOperators(Device::Type deviceType);
std::shared_ptr<Operators> getOperatorsSharedPtr(Device::Type deviceType);

bool isOperatorsAvailable(Device::Type deviceType);
void initOperators();
void destroyOperators();

}  // namespace fl
