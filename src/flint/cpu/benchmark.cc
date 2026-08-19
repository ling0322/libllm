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

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "catch2/catch_amalgamated.hpp"
#include "lutil/time.h"
#include "flint/device.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace {

constexpr int HiddenSize = 3072;
constexpr int IntermediateSize = 8192;
constexpr int NumHeads = 24;
constexpr int NumKeyValueHeads = 8;
constexpr int HeadDim = 128;
constexpr int QkvSize = HiddenSize + 2 * NumKeyValueHeads * HeadDim;
constexpr int VocabSize = 128256;
constexpr int BatchSize = 1;

// CPU clocks and OpenMP threads need time to settle, so measure against a time budget instead of
// a fixed repeat count.
constexpr double WarmupSeconds = 0.2;
constexpr double MeasureSeconds = 0.3;

template<typename Fn>
float benchmarkCpu(Fn &&fn) {
  double warmupBegin = lut::now();
  do {
    fn();
  } while (lut::now() - warmupBegin < WarmupSeconds);

  int numIterations = 0;
  double elapsed = 0.0;
  double begin = lut::now();
  do {
    fn();
    ++numIterations;
    elapsed = lut::now() - begin;
  } while (elapsed < MeasureSeconds);

  return static_cast<float>(elapsed * 1000.0 / numIterations);}

int getNumThreads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void printLatency(const std::string &name, float milliseconds) {
  std::printf("%-44s %10.3f ms\n", name.c_str(), milliseconds);
}

void printMatmul(const std::string &name, float milliseconds, int m, int n, int k) {
  double gflops = 2.0 * m * n * k / (milliseconds * 1.0e6);
  std::printf("%-44s %10.3f ms  %8.2f GFLOP/s\n", name.c_str(), milliseconds, gflops);
}

Tensor randFloat(const std::shared_ptr<Operators> &operators, std::initializer_list<int> shape) {
  return operators->rand(shape, DType::kFloat);
}

void benchmarkMatmul(
    const std::shared_ptr<Operators> &operators, const char *name, int m, int n, int k) {
  Tensor input = randFloat(operators, {m, k});
  Tensor weight = randFloat(operators, {n, k}).transpose(0, 1);
  float milliseconds = benchmarkCpu([&] { operators->matmul(input, weight); });
  printMatmul(name, milliseconds, m, n, k);
}

void benchmarkRmsNorm(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor input = randFloat(operators, {BatchSize, sequenceLength, HiddenSize});
  Tensor weight = randFloat(operators, {HiddenSize});
  float milliseconds = benchmarkCpu([&] { operators->rmsNorm(input, weight, 1.0e-5f); });
  printLatency("rms_norm [1," + std::to_string(sequenceLength) + ",3072]", milliseconds);
}

void benchmarkSwiGlu(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor input = randFloat(operators, {BatchSize, sequenceLength, 2 * IntermediateSize});
  float milliseconds = benchmarkCpu([&] { operators->swiglu(input); });
  printLatency("swiglu [1," + std::to_string(sequenceLength) + ",16384]", milliseconds);
}

void benchmarkResidualAdd(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor input = randFloat(operators, {BatchSize, sequenceLength, HiddenSize});
  Tensor residual = randFloat(operators, {BatchSize, sequenceLength, HiddenSize});
  float milliseconds = benchmarkCpu([&] { operators->add(input, residual); });
  printLatency("residual_add [1," + std::to_string(sequenceLength) + ",3072]", milliseconds);
}

void benchmarkSoftmax(const std::shared_ptr<Operators> &operators) {
  Tensor logits = randFloat(operators, {BatchSize, VocabSize});
  float milliseconds = benchmarkCpu([&] { operators->softmax(logits); });
  printLatency("softmax [1,128256]", milliseconds);
}

void benchmarkAttention(
    const std::shared_ptr<Operators> &operators, int queryLength, int keyValueLength) {
  Tensor q = randFloat(operators, {BatchSize, NumHeads, queryLength, HeadDim});
  Tensor k = randFloat(operators, {BatchSize, NumKeyValueHeads, keyValueLength, HeadDim});
  Tensor v = randFloat(operators, {BatchSize, NumKeyValueHeads, keyValueLength, HeadDim});

  bool causal = queryLength > 1;
  float milliseconds = benchmarkCpu([&] { operators->attention(q, k, v, causal); });
  printLatency(
      "attention [24," + std::to_string(queryLength) + "," + std::to_string(keyValueLength) +
          ",128]",
      milliseconds);
}

Tensor createTokenIds(int sequenceLength) {
  std::vector<LongType> values(sequenceLength);
  for (int i = 0; i < sequenceLength; ++i) values[i] = i % VocabSize;
  return Tensor::create<LongType>({BatchSize, sequenceLength}, values);
}

void benchmarkLookup(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor embedding = randFloat(operators, {VocabSize, HiddenSize});
  Tensor ids = createTokenIds(sequenceLength);
  float milliseconds = benchmarkCpu([&] { operators->lookup(embedding, ids); });
  printLatency("embedding_lookup [" + std::to_string(sequenceLength) + ",3072]", milliseconds);
}

}  // namespace

CATCH_TEST_CASE("Llama 3.2 3B CPU benchmarks", "[benchmark][cpu][llama32-3b]") {
  std::shared_ptr<Operators> operators = getOperatorsSharedPtr(Device::kCpu);

  std::printf("\nLlama 3.2 3B projection benchmarks (FP32, %d threads)\n", getNumThreads());
  for (int sequenceLength : {1, 128, 512}) {
    int m = BatchSize * sequenceLength;
    std::string prefix = sequenceLength == 1 ? "decode" : "prefill";
    prefix += "-" + std::to_string(sequenceLength);
    benchmarkMatmul(operators, (prefix + " qkv_proj").c_str(), m, QkvSize, HiddenSize);
    benchmarkMatmul(operators, (prefix + " out_proj").c_str(), m, HiddenSize, HiddenSize);
    benchmarkMatmul(
        operators, (prefix + " gate_up_proj").c_str(), m, 2 * IntermediateSize, HiddenSize);
    benchmarkMatmul(operators, (prefix + " down_proj").c_str(), m, HiddenSize, IntermediateSize);
  }
  benchmarkMatmul(operators, "decode-1 lm_head", 1, VocabSize, HiddenSize);

  std::printf("\nLlama 3.2 3B normalization and elementwise benchmarks (FP32)\n");
  for (int sequenceLength : {1, 128, 512}) {
    benchmarkRmsNorm(operators, sequenceLength);
    benchmarkSwiGlu(operators, sequenceLength);
    benchmarkResidualAdd(operators, sequenceLength);
  }

  std::printf("\nLlama 3.2 3B attention benchmarks (FP32)\n");
  benchmarkAttention(operators, 128, 128);
  benchmarkAttention(operators, 512, 512);
  benchmarkAttention(operators, 1, 512);
  benchmarkAttention(operators, 1, 2048);

  std::printf("\nLlama 3.2 3B embedding and generation benchmarks (FP32)\n");
  benchmarkLookup(operators, 1);
  benchmarkLookup(operators, 128);

  Tensor logits = randFloat(operators, {BatchSize, VocabSize});
  Tensor history = createTokenIds(32);
  float milliseconds = benchmarkCpu([&] { operators->repetitionPenalty(logits, history, 1.1f); });
  printLatency("repetition_penalty [1,128256] history=32", milliseconds);
  benchmarkSoftmax(operators);
}

}  // namespace fl
