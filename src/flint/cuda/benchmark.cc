#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/cuda/common.h"
#include "flint/cuda/cuda_operators.h"

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

class CudaEvent {
 public:
	CudaEvent() {
		LL_CHECK_CUDA_STATUS(cudaEventCreate(&_event));
	}

	~CudaEvent() {
		cudaEventDestroy(_event);
	}

	operator cudaEvent_t() const {
		return _event;
	}

 private:
	cudaEvent_t _event;
};

template<typename Fn>
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

std::shared_ptr<Operators> createCudaOperators() {
	return op::cuda::CudaOperators::create();
}

Tensor randHalf(const std::shared_ptr<Operators> &operators, std::initializer_list<int> shape) {
	return operators->rand(shape, DType::kFloat16);
}

void benchmarkMatmul(
		const std::shared_ptr<Operators> &operators,
		const char *name,
		int m,
		int n,
		int k) {
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

void benchmarkSoftmax(
		const std::shared_ptr<Operators> &operators,
		int queryLength,
		int keyValueLength) {
	Tensor scores = randHalf(operators, {BatchSize, NumHeads, queryLength, keyValueLength});
	float milliseconds = benchmarkCuda([&] { operators->softmax(scores); });
	printLatency(
			"softmax [1,24," + std::to_string(queryLength) + "," +
					std::to_string(keyValueLength) + "]",
			milliseconds);
}

		void benchmarkAttentionMatmul(
			const std::shared_ptr<Operators> &operators,
			int queryLength,
			int keyValueLength) {
		  Tensor q = randHalf(operators, {BatchSize, NumHeads, queryLength, HeadDim});
		  Tensor k = randHalf(operators, {BatchSize, NumHeads, keyValueLength, HeadDim});
		  Tensor scores = randHalf(operators, {BatchSize, NumHeads, queryLength, keyValueLength});
		  Tensor v = randHalf(operators, {BatchSize, NumHeads, keyValueLength, HeadDim});

		  float qkMilliseconds = benchmarkCuda([&] { operators->matmul(q, k.transpose(-2, -1)); });
		  printLatency(
			  "attention_qk [24," + std::to_string(queryLength) + "," +
				  std::to_string(keyValueLength) + ",128]",
			  qkMilliseconds);

		  float pvMilliseconds = benchmarkCuda([&] { operators->matmul(scores, v); });
		  printLatency(
			  "attention_pv [24," + std::to_string(queryLength) + "," +
				  std::to_string(keyValueLength) + ",128]",
			  pvMilliseconds);
		}

void benchmarkGqaCopy(const std::shared_ptr<Operators> &operators, int sequenceLength) {
	constexpr int groupSize = NumHeads / NumKeyValueHeads;
	Tensor input = randHalf(operators, {BatchSize, sequenceLength, NumKeyValueHeads, 1, HeadDim});
	input = input.expand({BatchSize, sequenceLength, NumKeyValueHeads, groupSize, HeadDim});
	Tensor output = operators->tensorLike(input);
	float milliseconds = benchmarkCuda([&] { operators->copy(input, output); });
	printLatency(
			"gqa_copy [1," + std::to_string(sequenceLength) + ",8,3,128]",
			milliseconds);
}

Tensor createTokenIds(const std::shared_ptr<Operators> &operators, int sequenceLength) {
	std::vector<LongType> values(sequenceLength);
	for (int i = 0; i < sequenceLength; ++i) values[i] = i % VocabSize;
	Tensor cpuIds = Tensor::create<LongType>({BatchSize, sequenceLength}, values);
	return operators->to(Device::getCuda(), cpuIds);
}

void benchmarkLookup(const std::shared_ptr<Operators> &operators, int sequenceLength) {
	Tensor embedding = randHalf(operators, {VocabSize, HiddenSize});
	Tensor ids = createTokenIds(operators, sequenceLength);
	float milliseconds = benchmarkCuda([&] { operators->lookup(embedding, ids); });
	printLatency(
			"embedding_lookup [" + std::to_string(sequenceLength) + ",3072]",
			milliseconds);
}

void benchmarkSampling(const std::shared_ptr<Operators> &operators) {
	Tensor logits = randHalf(operators, {BatchSize, VocabSize});
	Tensor distribution = operators->softmax(logits);
	float milliseconds = benchmarkCuda([&] { operators->sample(distribution, 50, 0.9f); });
	printLatency("sampling [1,128256] top_k=50 top_p=0.9", milliseconds);
}

}  // namespace

CATCH_TEST_CASE("Llama 3.2 3B projection benchmarks", "[benchmark][cuda][llama32-3b][matmul]") {
	if (!op::cuda::CudaOperators::isAvailable()) CATCH_SKIP("cuda device not available");
	std::shared_ptr<Operators> operators = createCudaOperators();

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
}

CATCH_TEST_CASE("Llama 3.2 3B elementwise benchmarks", "[benchmark][cuda][llama32-3b][elementwise]") {
	if (!op::cuda::CudaOperators::isAvailable()) CATCH_SKIP("cuda device not available");
	std::shared_ptr<Operators> operators = createCudaOperators();

	std::printf("\nLlama 3.2 3B normalization and elementwise benchmarks (FP16)\n");
	for (int sequenceLength : {1, 128, 512}) {
		benchmarkRmsNorm(operators, sequenceLength);
		benchmarkSwiGlu(operators, sequenceLength);
		benchmarkResidualAdd(operators, sequenceLength);
	}
}

CATCH_TEST_CASE("Llama 3.2 3B attention benchmarks", "[benchmark][cuda][llama32-3b][attention]") {
	if (!op::cuda::CudaOperators::isAvailable()) CATCH_SKIP("cuda device not available");
	std::shared_ptr<Operators> operators = createCudaOperators();

	std::printf("\nLlama 3.2 3B attention primitive benchmarks (FP16)\n");
	benchmarkSoftmax(operators, 128, 128);
	benchmarkSoftmax(operators, 512, 512);
	benchmarkSoftmax(operators, 1, 512);
	benchmarkSoftmax(operators, 1, 2048);
	benchmarkAttentionMatmul(operators, 128, 128);
	benchmarkAttentionMatmul(operators, 512, 512);
	benchmarkAttentionMatmul(operators, 1, 512);
	benchmarkAttentionMatmul(operators, 1, 2048);
}

CATCH_TEST_CASE("Llama 3.2 3B layout benchmarks", "[benchmark][cuda][llama32-3b][layout]") {
	if (!op::cuda::CudaOperators::isAvailable()) CATCH_SKIP("cuda device not available");
	std::shared_ptr<Operators> operators = createCudaOperators();

	std::printf("\nLlama 3.2 3B layout benchmarks (FP16)\n");
	benchmarkGqaCopy(operators, 128);
	benchmarkGqaCopy(operators, 512);
	for (int sequenceLength : {128, 512}) {
		float milliseconds = benchmarkCuda([&] { operators->causalMask(sequenceLength); });
		printLatency("causal_mask [" + std::to_string(sequenceLength) + "," +
										 std::to_string(sequenceLength) + "]",
								 milliseconds);
	}
}

CATCH_TEST_CASE("Llama 3.2 3B generation benchmarks", "[benchmark][cuda][llama32-3b][generation]") {
	if (!op::cuda::CudaOperators::isAvailable()) CATCH_SKIP("cuda device not available");
	std::shared_ptr<Operators> operators = createCudaOperators();

	std::printf("\nLlama 3.2 3B embedding and generation benchmarks (FP16)\n");
	benchmarkLookup(operators, 1);
	benchmarkLookup(operators, 128);

	Tensor logits = randHalf(operators, {BatchSize, VocabSize});
	Tensor history = createTokenIds(operators, 32);
	float milliseconds = benchmarkCuda([&] { operators->repetitionPenalty(logits, history, 1.1f); });
	printLatency("repetition_penalty [1,128256] history=32", milliseconds);
	benchmarkSampling(operators);
}

}  // namespace fl
