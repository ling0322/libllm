#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/cuda/common.h"
#include "flint/cuda/cuda_operators.h"
#include "flint/cuda/gated_delta_net.h"
#include "flint/functional.h"
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
constexpr int NumWarmup = 5;
constexpr int NumIterations = 20;

// The Qwen3.5 gated DeltaNet layer: 16 key heads against 48 value heads, both 128 wide. 48 of the
// model's 64 layers run this rather than the attention the rest of these benchmarks cover.
constexpr int DeltaNetKeyHeads = 16;
constexpr int DeltaNetValueHeads = 48;
constexpr int DeltaNetHeadDim = 128;

class CudaEvent {
 public:
  CudaEvent() { LL_CHECK_CUDA_STATUS(cudaEventCreate(&_event)); }

  ~CudaEvent() { cudaEventDestroy(_event); }

  operator cudaEvent_t() const { return _event; }

 private:
  cudaEvent_t _event;
};

template <typename Fn>
float benchmarkCuda(Fn &&fn) {
  for (int i = 0; i < NumWarmup; ++i) fn();
  LL_CHECK_CUDA_STATUS(cudaDeviceSynchronize());

  CudaEvent begin;
  CudaEvent end;
  LL_CHECK_CUDA_STATUS(cudaEventRecord(begin));
  for (int i = 0; i < NumIterations; ++i) fn();
  LL_CHECK_CUDA_STATUS(cudaEventRecord(end));
  LL_CHECK_CUDA_STATUS(cudaEventSynchronize(end));

  float totalMs;
  LL_CHECK_CUDA_STATUS(cudaEventElapsedTime(&totalMs, begin, end));
  return totalMs / NumIterations;
}

void printLatency(const std::string &name, float milliseconds) {
  std::printf("%-44s %10.3f us\n", name.c_str(), milliseconds * 1000.0f);
}

void printMatmul(const std::string &name, float milliseconds, int m, int n, int k) {
  double tflops = 2.0 * m * n * k / (milliseconds * 1.0e9);
  std::printf("%-44s %10.3f us  %8.2f TFLOP/s\n", name.c_str(), milliseconds * 1000.0f, tflops);
}

Tensor randHalf(const std::shared_ptr<Operators> &operators, std::initializer_list<int> shape) {
  return operators->rand(shape, DType::kFloat16);
}

void benchmarkMatmul(
    const std::shared_ptr<Operators> &operators, const char *name, int m, int n, int k) {
  Tensor input = randHalf(operators, {m, k});
  Tensor weight = randHalf(operators, {n, k}).transpose(0, 1);
  float milliseconds = benchmarkCuda([&] { operators->matmul(input, weight); });
  printMatmul(name, milliseconds, m, n, k);
}

void benchmarkRmsNorm(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor input = randHalf(operators, {BatchSize, sequenceLength, HiddenSize});
  Tensor weight = randHalf(operators, {HiddenSize});
  float milliseconds = benchmarkCuda([&] { operators->rmsNorm(input, weight, 1.0e-5f); });
  printLatency("rms_norm [1," + std::to_string(sequenceLength) + ",3072]", milliseconds);
}

void benchmarkSwiGlu(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor input = randHalf(operators, {BatchSize, sequenceLength, 2 * IntermediateSize});
  float milliseconds = benchmarkCuda([&] { operators->swiglu(input); });
  printLatency("swiglu [1," + std::to_string(sequenceLength) + ",16384]", milliseconds);
}

void benchmarkResidualAdd(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor input = randHalf(operators, {BatchSize, sequenceLength, HiddenSize});
  Tensor residual = randHalf(operators, {BatchSize, sequenceLength, HiddenSize});
  float milliseconds = benchmarkCuda([&] { operators->add(input, residual); });
  printLatency("residual_add [1," + std::to_string(sequenceLength) + ",3072]", milliseconds);
}

void benchmarkSoftmax(const std::shared_ptr<Operators> &operators) {
  Tensor logits = randHalf(operators, {BatchSize, VocabSize});
  float milliseconds = benchmarkCuda([&] { operators->softmax(logits); });
  printLatency("softmax [1,128256]", milliseconds);
}

void benchmarkAttention(
    const std::shared_ptr<Operators> &operators, int queryLength, int keyValueLength) {
  Tensor q = randHalf(operators, {BatchSize, NumHeads, queryLength, HeadDim});
  Tensor k = randHalf(operators, {BatchSize, NumKeyValueHeads, keyValueLength, HeadDim});
  Tensor v = randHalf(operators, {BatchSize, NumKeyValueHeads, keyValueLength, HeadDim});

  bool causal = queryLength > 1;
  float milliseconds = benchmarkCuda([&] { operators->attention(q, k, v, causal); });
  printLatency(
      "attention [24," + std::to_string(queryLength) + "," + std::to_string(keyValueLength) +
          ",128]",
      milliseconds);
}

Tensor createTokenIds(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  std::vector<LongType> values(sequenceLength);
  for (int i = 0; i < sequenceLength; ++i) values[i] = i % VocabSize;
  Tensor cpuIds = Tensor::create<LongType>({BatchSize, sequenceLength}, values);
  return operators->to(Device::getCuda(), cpuIds);
}

Tensor createPositions(const std::shared_ptr<Operators> &operators, int numTokens) {
  std::vector<LongType> values(numTokens);
  for (int i = 0; i < numTokens; ++i) values[i] = i;
  Tensor cpuPositions = Tensor::create<LongType>({numTokens}, values);
  return operators->to(Device::getCuda(), cpuPositions);
}

Tensor applyRotaryEmbeddingBaseline(
    const std::shared_ptr<Operators> &operators,
    Tensor input,
    Tensor roPE) {
  Tensor cos = roPE.subtensor(0);
  Tensor sin = roPE.subtensor(1);
  cos = cos.expand({cos.getShape(0), input.getShape(1), cos.getShape(2)});
  sin = sin.expand({sin.getShape(0), input.getShape(1), sin.getShape(2)});

  int halfShape = input.getShape(-1) / 2;
  Tensor rotated = operators->tensorLike(input);
  Tensor x1 = input.slice(-1, {0, halfShape});
  Tensor x2 = operators->mul(input.slice(-1, {halfShape, None}), -1.0f);
  operators->copy(x1, rotated.slice(-1, {halfShape, None}));
  operators->copy(x2, rotated.slice(-1, {0, halfShape}));

  return operators->add(
      operators->mul(input, F::contiguous(cos)),
      operators->mul(rotated, F::contiguous(sin)));
}

std::pair<Tensor, Tensor> rotaryEmbeddingBaseline(
    const std::shared_ptr<Operators> &operators,
    Tensor positions,
    Tensor query,
    Tensor key,
    Tensor rotaryCache) {
  int numTokens = positions.getShape(0);
  Tensor roPE = operators->lookup(rotaryCache, positions);
  roPE = roPE.view({numTokens, 2, 1, HeadDim}).transpose(0, 1);
  return {
      applyRotaryEmbeddingBaseline(operators, query, roPE),
      applyRotaryEmbeddingBaseline(operators, key, roPE)};
}

void benchmarkRotaryEmbedding(
    const std::shared_ptr<Operators> &operators,
    int numTokens) {
  constexpr int MaxPositions = 8192;
  Tensor positions = createPositions(operators, numTokens);
  Tensor query = randHalf(operators, {numTokens, NumHeads, HeadDim});
  Tensor key = randHalf(operators, {numTokens, NumKeyValueHeads, HeadDim});
  Tensor rotaryCache = randHalf(operators, {MaxPositions, 2 * HeadDim});

  float baselineMilliseconds = benchmarkCuda([&] {
    auto output = rotaryEmbeddingBaseline(operators, positions, query, key, rotaryCache);
    (void)output;
  });
  Tensor fusedQuery = F::contiguous(query);
  Tensor fusedKey = F::contiguous(key);
  float fusedMilliseconds = benchmarkCuda([&] {
    operators->rotaryEmbedding(positions, fusedQuery, fusedKey, rotaryCache);
  });

  std::string phase = numTokens == 1 ? "decode-1" : "prefill-" + std::to_string(numTokens);
  printLatency(
      phase + " rotary_embedding baseline [24/8,128]",
      baselineMilliseconds);
  printLatency(
      phase + " rotary_embedding fused [24/8,128]",
      fusedMilliseconds);
  std::printf(
      "%-44s %10.2fx\n",
      (phase + " rotary_embedding speedup").c_str(),
      baselineMilliseconds / fusedMilliseconds);
}

void benchmarkLookup(const std::shared_ptr<Operators> &operators, int sequenceLength) {
  Tensor embedding = randHalf(operators, {VocabSize, HiddenSize});
  Tensor ids = createTokenIds(operators, sequenceLength);
  float milliseconds = benchmarkCuda([&] { operators->lookup(embedding, ids); });
  printLatency("embedding_lookup [" + std::to_string(sequenceLength) + ",3072]", milliseconds);
}

void benchmarkSampling(const std::shared_ptr<Operators> &operators) {
  Tensor logits = randHalf(operators, {BatchSize, VocabSize});
  Tensor temperatures = operators->to(
    Device::getCuda(), Tensor::create<float>({BatchSize}, {1.0f}));
  Tensor topKs = operators->to(
    Device::getCuda(), Tensor::create<IntType>({BatchSize}, {50}));
  Tensor topPs = operators->to(
    Device::getCuda(), Tensor::create<float>({BatchSize}, {0.9f}));
  float milliseconds = benchmarkCuda(
    [&] { operators->sample(logits, temperatures, topKs, topPs); });
  printLatency("sampling [1,128256] top_k=50 top_p=0.9", milliseconds);
}

const char *pathName(op::cuda::GatedDeltaNetPath path) {
  switch (path) {
    case op::cuda::GatedDeltaNetPath::kTensorCoreMma:
      return "mma";
    case op::cuda::GatedDeltaNetPath::kTensorCoreMmaChunkOnly:
      return "mma-chunk-only";
    default:
      return "auto";
  }
}

void benchmarkGatedDeltaNetSeqlens(
    const std::shared_ptr<Operators> &operators,
    const std::vector<int> &seqlens,
    const std::string &label,
    op::cuda::GatedDeltaNetPath path) {
  int numSeq = static_cast<int>(seqlens.size());
  int numTokens = 0;
  for (int len : seqlens) numTokens += len;

  Tensor q = randHalf(operators, {numTokens, DeltaNetKeyHeads, DeltaNetHeadDim});
  Tensor k = randHalf(operators, {numTokens, DeltaNetKeyHeads, DeltaNetHeadDim});
  Tensor v = randHalf(operators, {numTokens, DeltaNetValueHeads, DeltaNetHeadDim});
  Tensor g = operators->rand({numTokens, DeltaNetValueHeads}, DType::kFloat);
  Tensor beta = operators->rand({numTokens, DeltaNetValueHeads}, DType::kFloat);
  // CudaOperators::zeros hands back a <half> whatever dtype it is asked for, so the state starts
  // as a host tensor and is copied over.
  std::vector<float> stateData(
      static_cast<size_t>(numSeq) * DeltaNetValueHeads * DeltaNetHeadDim * DeltaNetHeadDim,
      0.0f);
  Tensor state = F::to(
      Device::getCuda(),
      Tensor::create<float>(
          {numSeq, DeltaNetValueHeads, DeltaNetHeadDim, DeltaNetHeadDim},
          lut::makeConstSpan(stateData)));

  // g is a log decay, so it has to be at most zero.
  g = operators->neg(g);

  std::vector<int32_t> lengths;
  lengths.push_back(0);
  for (int len : seqlens) lengths.push_back(lengths.back() + len);
  Tensor cuSeqlens = F::to(
      Device::getCuda(),
      Tensor::create<int32_t>({numSeq + 1}, lut::makeConstSpan(lengths)));

  // The identity mapping: a pool the size of the batch, in batch order. What the mapping costs is
  // one int load per block, so a scattered pool measures the same; it is the timings that are
  // being kept comparable here, not the traffic.
  std::vector<int32_t> slots;
  for (int s = 0; s < numSeq; ++s) slots.push_back(s);
  Tensor stateSlots = F::to(
      Device::getCuda(),
      Tensor::create<int32_t>({numSeq}, lut::makeConstSpan(slots)));

  float milliseconds = benchmarkCuda([&] {
    op::cuda::gatedDeltaNetPrefill(q, k, v, g, beta, cuSeqlens, stateSlots, state, path);
  });
  printLatency(std::string("gated_delta_net ") + pathName(path) + " " + label, milliseconds);
}

// The even split, which is what the prefill benchmarks measure: one batch of `numSeq` sequences
// that between them hold `numTokens`.
void benchmarkGatedDeltaNet(
    const std::shared_ptr<Operators> &operators,
    int numTokens,
    int numSeq,
    op::cuda::GatedDeltaNetPath path) {
  std::vector<int> seqlens(numSeq, numTokens / numSeq);
  seqlens.back() += numTokens - numSeq * (numTokens / numSeq);
  benchmarkGatedDeltaNetSeqlens(
      operators,
      seqlens,
      "tokens=" + std::to_string(numTokens) + " seqs=" + std::to_string(numSeq),
      path);
}

}  // namespace

CATCH_TEST_CASE("Qwen3.5 gated DeltaNet benchmarks", "[benchmark][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");
  std::shared_ptr<Operators> operators = getOperatorsSharedPtr(Device::kCuda);

  std::printf("\nQwen3.5 gated DeltaNet prefill benchmarks (FP16 in, FP32 state)\n");

  // One CTA per (sequence, value head), so a single sequence gives the kernel only 48 of them
  // against 36 SMs. Splitting the same token count over several sequences is what a serving batch
  // looks like, and is what the launch's shape actually turns on.
  for (int numSeq : {1, 2, 4, 8}) {
    benchmarkGatedDeltaNet(operators, 4096, numSeq, op::cuda::GatedDeltaNetPath::kTensorCoreMma);
  }

  for (int numTokens : {256, 1024, 4096}) {
    benchmarkGatedDeltaNet(operators, numTokens, 1, op::cuda::GatedDeltaNetPath::kAuto);
  }

  // A decode step, and a decode step with one sequence still prefilling in it. Both are what a
  // continuously batched server spends most of its launches on, and both are what the mma path's
  // per-CTA branch is for: mma-chunk-only is the same kernel with that branch turned off, so the
  // pair of them is the branch's whole cost and benefit.
  std::printf("\nQwen3.5 gated DeltaNet decode benchmarks\n");
  for (op::cuda::GatedDeltaNetPath path :
       {op::cuda::GatedDeltaNetPath::kTensorCoreMma,
        op::cuda::GatedDeltaNetPath::kTensorCoreMmaChunkOnly}) {
    for (int numSeq : {1, 8, 32, 128}) {
      benchmarkGatedDeltaNetSeqlens(
          operators,
          std::vector<int>(numSeq, 1),
          "decode seqs=" + std::to_string(numSeq),
          path);
    }

    for (int numSeq : {32, 128}) {
      std::vector<int> seqlens(numSeq, 1);
      seqlens.back() = 2048;
      benchmarkGatedDeltaNetSeqlens(
          operators,
          seqlens,
          "decode seqs=" + std::to_string(numSeq) + " + one 2048 prefill",
          path);
    }
  }
}

CATCH_TEST_CASE("CUDA rotary embedding benchmarks", "[benchmark][cuda][rope]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");
  std::shared_ptr<Operators> operators = getOperatorsSharedPtr(Device::kCuda);

  std::printf("\nLlama 3.2 3B rotary embedding baseline (FP16)\n");
  for (int numTokens : {1, 256, 2048}) {
    benchmarkRotaryEmbedding(operators, numTokens);
  }
}

CATCH_TEST_CASE("Llama 3.2 3B benchmarks", "[benchmark][cuda][llama32-3b]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");
  std::shared_ptr<Operators> operators = getOperatorsSharedPtr(Device::kCuda);

  std::printf("\nLlama 3.2 3B projection benchmarks (FP16)\n");
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

  std::printf("\nLlama 3.2 3B normalization and elementwise benchmarks (FP16)\n");
  for (int sequenceLength : {1, 128, 512}) {
    benchmarkRmsNorm(operators, sequenceLength);
    benchmarkSwiGlu(operators, sequenceLength);
    benchmarkResidualAdd(operators, sequenceLength);
  }

  std::printf("\nLlama 3.2 3B attention benchmarks (FP16)\n");
  benchmarkAttention(operators, 128, 128);
  benchmarkAttention(operators, 512, 512);
  benchmarkAttention(operators, 1, 512);
  benchmarkAttention(operators, 1, 2048);

  std::printf("\nLlama 3.2 3B embedding and generation benchmarks (FP16)\n");
  benchmarkLookup(operators, 1);
  benchmarkLookup(operators, 128);

  Tensor logits = randHalf(operators, {BatchSize, VocabSize});
  Tensor history = createTokenIds(operators, 32);
  float milliseconds = benchmarkCuda([&] { operators->repetitionPenalty(logits, history, 1.1f); });
  printLatency("repetition_penalty [1,128256] history=32", milliseconds);
  benchmarkSoftmax(operators);
  benchmarkSampling(operators);
}

}  // namespace fl
