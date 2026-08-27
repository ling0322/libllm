// The MIT License (MIT)
//
// Copyright (c) 2023-2025 Xiaoyang Chen
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

#include "flint/operators.h"

namespace fl {
namespace op {
namespace cuda {

class MatMul;
class Rand;

/// @brief Implementation of Operator interface with cuda device.
class CudaOperators : public Operators {
 public:
  ~CudaOperators() = default;

  static constexpr int OPT_CUTLASS_GEMM = 0x00000001;
  static constexpr int OPT_CUBLAS_GEMM = 0x00000002;

  /// @brief Returns true if the CudaOperators is available (CUDA device available in host).
  /// @return if CudaOperators available.
  static bool isAvailable();

  // create a instance of CudaOperators
  static std::shared_ptr<Operators> create(int options = 0);

  // implement interface Operators
  Tensor arangeLong(LongType begin, LongType end, LongType step) override;
  Tensor cast(Tensor tensor, DType dtype) override;
  Tensor add(Tensor a, Tensor b) override;
  Tensor sub(Tensor input, Tensor other) override;
  Tensor causalMask(int max_len) override;
  void copy(Tensor src, Tensor dest) override;
  void fill(Tensor input, float value) override;
  Tensor square(Tensor input) override;
  Tensor lookup(Tensor table, Tensor indices) override;
  void rotaryEmbedding(Tensor positions, Tensor query, Tensor key, Tensor rotaryCache) override;
  Tensor matmul(Tensor a, Tensor b) override;
  Tensor matmulNarrowPrecision(Tensor A, Tensor sfA, Tensor B, Tensor sfB) override;
  Tensor gatedDeltaNetPrefill(
      Tensor q,
      Tensor k,
      Tensor v,
      Tensor g,
      Tensor beta,
      Tensor cuSeqlens,
      Tensor stateSlots,
      Tensor state) override;
  Tensor max(Tensor inputs) override;
  Tensor min(Tensor inputs) override;
  Tensor divTensor(Tensor input, Tensor other) override;
  Tensor neg(Tensor input) override;
  Tensor abs(Tensor input) override;
  Tensor exp(Tensor input) override;
  Tensor sqrt(Tensor input) override;
  Tensor rsqrt(Tensor input) override;
  Tensor sigmoid(Tensor input) override;
  Tensor tanh(Tensor input) override;
  Tensor relu(Tensor input) override;
  Tensor gelu(Tensor input) override;
  Tensor silu(Tensor input) override;
  Tensor subFloat(Tensor input, float other) override;
  bool allClose(Tensor A, Tensor B, float rtol, float atol) override;
  Tensor mul(Tensor input, float other) override;
  bool all(Tensor A) override;
  Tensor div(Tensor input, float other) override;
  Tensor mod(Tensor input, LongType other) override;
  Tensor mul(Tensor input, Tensor other) override;
  void print(Tensor tensor) override;
  void repetitionPenalty(Tensor logits, Tensor history, float weight) override;
  Tensor rmsNorm(Tensor input, Tensor weight, float eps) override;
  Tensor sample(Tensor logits, Tensor temperatures, Tensor topKs, Tensor topPs) override;
  Tensor softmax(Tensor input) override;
  Tensor attention(Tensor q, Tensor k, Tensor v, bool causal) override;
    Tensor pagedAttention(
      Tensor q,
      Tensor keyCache,
      Tensor valueCache,
      Tensor blockTable,
      Tensor cuSeqlensQ,
      Tensor seqlensK,
      int maxQLen,
      int maxKLen,
      bool causal) override;
    void storeKVCache(
      Tensor k,
      Tensor v,
      Tensor keyCache,
      Tensor valueCache,
      Tensor slotMapping) override;
  Tensor sum(Tensor inputs, int dim) override;
  Tensor swiglu(Tensor A) override;
  Tensor tensor(lut::Span<const int> shape, DType dtype) override;
  Tensor tensorLike(Tensor input) override;
  Tensor to(Device device, Tensor tensor) override;
  Tensor zeros(lut::Span<const int> shape, DType dtype) override;
  Tensor randNormal(lut::Span<const int> shape) override;
  Tensor rand(lut::Span<const int> shape, DType dtype) override;
  void manualSeed(uint64_t seed) override;
  float elem(Tensor tensor) override;
  bool elemBool(Tensor tensor) override;
  Tensor eq(Tensor input, Tensor other) override;

  MemorySnapshot captureMemorySnapshot() override;
  void resetPeakMemoryStats() override;

  DType getDefaultFloatType() override;

 private:
  std::shared_ptr<MatMul> _matmul;
  std::shared_ptr<Rand> _rand;

  CudaOperators() = default;
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
