#include "flint/cuda/sampling.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {

namespace {

constexpr int BlockSize = 256;
constexpr int RadixBits = 2;
constexpr int RadixBuckets = 1 << RadixBits;
constexpr int SharedTopK = 2048;
constexpr int SharedItemsPerThread = SharedTopK / BlockSize;

template<typename T>
__device__ float normalizedLogit(T value) {
  float logit = static_cast<float>(value);
  return isnan(logit) ? -INFINITY : logit;
}

__device__ uint32_t orderedFloatKey(float value) {
  uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

__device__ int effectiveTopK(float temperature, IntType topK, int vocabSize) {
  if (temperature == 0.0f) return 1;
  return topK <= 0 ? vocabSize : min(static_cast<int>(topK), vocabSize);
}

__device__ float samplingWeight(float logit, float maxLogit, float temperature) {
  if (isinf(maxLogit)) return logit == maxLogit ? 1.0f : 0.0f;
  return expf((logit - maxLogit) / temperature);
}

template<typename T>
__global__ void radixSelectTopKThresholdKernel(
    const T *__restrict__ logits,
    const float *__restrict__ temperatures,
    const IntType *__restrict__ topKs,
    uint32_t *__restrict__ thresholds,
    int rows,
    int vocabSize) {
  int row = blockIdx.x;
  if (row >= rows) return;

  __shared__ int histogram[RadixBuckets];
  __shared__ int rank;
  __shared__ uint32_t selectedPrefix;
  __shared__ uint32_t prefixMask;

  if (threadIdx.x == 0) {
    rank = effectiveTopK(temperatures[row], topKs[row], vocabSize);
    selectedPrefix = 0;
    prefixMask = 0;
  }
  __syncthreads();

  const T *rowLogits = logits + static_cast<int64_t>(row) * vocabSize;
  for (int shift = 32 - RadixBits; shift >= 0; shift -= RadixBits) {
    if (threadIdx.x < RadixBuckets) histogram[threadIdx.x] = 0;
    __syncthreads();

    int localHistogram[RadixBuckets] = {};
    for (int label = threadIdx.x; label < vocabSize; label += BlockSize) {
      uint32_t key = orderedFloatKey(normalizedLogit(rowLogits[label]));
      if ((key & prefixMask) == selectedPrefix) {
        ++localHistogram[(key >> shift) & (RadixBuckets - 1)];
      }
    }

#pragma unroll
    for (int bucket = 0; bucket < RadixBuckets; ++bucket) {
      if (localHistogram[bucket] != 0) atomicAdd(&histogram[bucket], localHistogram[bucket]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      for (int bucket = RadixBuckets - 1; bucket >= 0; --bucket) {
        if (rank > histogram[bucket]) {
          rank -= histogram[bucket];
        } else {
          selectedPrefix |= static_cast<uint32_t>(bucket) << shift;
          prefixMask |= static_cast<uint32_t>(RadixBuckets - 1) << shift;
          break;
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) thresholds[row] = selectedPrefix;
}

template<typename T>
__global__ void gatherTopKCandidatesKernel(
    const T *__restrict__ logits,
    const float *__restrict__ temperatures,
    const IntType *__restrict__ topKs,
    const uint32_t *__restrict__ thresholds,
    float *__restrict__ candidateLogits,
    int *__restrict__ candidateLabels,
    int rows,
    int vocabSize) {
  int row = blockIdx.x;
  if (row >= rows) return;

  using FlagScan = cub::BlockScan<int, BlockSize>;
  __shared__ typename FlagScan::TempStorage scanStorage;
  __shared__ int outputCount;

  if (threadIdx.x == 0) outputCount = 0;
  __syncthreads();

  int topK = effectiveTopK(temperatures[row], topKs[row], vocabSize);
  uint32_t threshold = thresholds[row];
  const T *rowLogits = logits + static_cast<int64_t>(row) * vocabSize;
  int64_t outputBase = static_cast<int64_t>(row) * vocabSize;

  for (int phase = 0; phase < 2; ++phase) {
    for (int begin = 0; begin < vocabSize && outputCount < topK; begin += BlockSize) {
      int label = begin + threadIdx.x;
      float logit = label < vocabSize ? normalizedLogit(rowLogits[label]) : -INFINITY;
      uint32_t key = orderedFloatKey(logit);
      int selected = label < vocabSize &&
          (phase == 0 ? key > threshold : key == threshold);
      int offset;
      int selectedInBlock;
      FlagScan(scanStorage).ExclusiveSum(selected, offset, selectedInBlock);

      int destination = outputCount + offset;
      if (selected && destination < topK) {
        candidateLogits[outputBase + destination] = logit;
        candidateLabels[outputBase + destination] = label;
      }
      __syncthreads();
      if (threadIdx.x == 0) outputCount = min(topK, outputCount + selectedInBlock);
      __syncthreads();
    }
  }
}

__global__ void initializeCandidateOffsetsKernel(
    const float *__restrict__ temperatures,
    const IntType *__restrict__ topKs,
    int *__restrict__ beginOffsets,
    int *__restrict__ endOffsets,
    int rows,
    int vocabSize) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  int begin = row * vocabSize;
  beginOffsets[row] = begin;
  endOffsets[row] = begin + effectiveTopK(temperatures[row], topKs[row], vocabSize);
}

__global__ void sampleTopPFromSortedCandidatesKernel(
    const float *__restrict__ candidateLogits,
    const int *__restrict__ candidateLabels,
    const float *__restrict__ uniformNoise,
    const float *__restrict__ temperatures,
    const IntType *__restrict__ topKs,
    const float *__restrict__ topPs,
    int rows,
    int vocabSize,
    LongType *__restrict__ result) {
  int row = blockIdx.x;
  if (row >= rows) return;

  int topK = effectiveTopK(temperatures[row], topKs[row], vocabSize);
  int64_t candidateOffset = static_cast<int64_t>(row) * vocabSize;
  if (temperatures[row] == 0.0f) {
    if (threadIdx.x == 0) result[row] = candidateLabels[candidateOffset];
    return;
  }

  float maxLogit = candidateLogits[candidateOffset];
  float uniform = uniformNoise[row];
  if (!isfinite(uniform) || uniform <= 0.0f) uniform = FLT_MIN;
  uniform = fminf(uniform, 0x1.fffffep-1f);

  if (topK <= SharedTopK) {
    __shared__ float prefixWeights[SharedTopK];
    __shared__ int selectedCount;
    __shared__ int sampledIndex;

    for (int index = threadIdx.x; index < topK; index += BlockSize) {
      prefixWeights[index] = samplingWeight(
          candidateLogits[candidateOffset + index], maxLogit, temperatures[row]);
    }
    __syncthreads();

    for (int offset = 1; offset < topK; offset <<= 1) {
      float additions[SharedItemsPerThread];
#pragma unroll
      for (int item = 0; item < SharedItemsPerThread; ++item) {
        int index = threadIdx.x + item * BlockSize;
        additions[item] = index < topK && index >= offset
            ? prefixWeights[index - offset]
            : 0.0f;
      }
      __syncthreads();
#pragma unroll
      for (int item = 0; item < SharedItemsPerThread; ++item) {
        int index = threadIdx.x + item * BlockSize;
        if (index < topK) prefixWeights[index] += additions[item];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) selectedCount = topK;
    __syncthreads();
    float topPThreshold = topPs[row] * prefixWeights[topK - 1];
    for (int index = threadIdx.x; index < topK; index += BlockSize) {
      if (prefixWeights[index] >= topPThreshold) atomicMin(&selectedCount, index + 1);
    }
    __syncthreads();

    float draw = uniform * prefixWeights[selectedCount - 1];
    if (threadIdx.x == 0) sampledIndex = selectedCount - 1;
    __syncthreads();
    for (int index = threadIdx.x; index < selectedCount; index += BlockSize) {
      if (prefixWeights[index] > draw) atomicMin(&sampledIndex, index);
    }
    __syncthreads();
    if (threadIdx.x == 0) result[row] = candidateLabels[candidateOffset + sampledIndex];
    return;
  }

  if (threadIdx.x == 0) {
    float totalWeight = 0.0f;
    for (int index = 0; index < topK; ++index) {
      totalWeight += samplingWeight(
          candidateLogits[candidateOffset + index], maxLogit, temperatures[row]);
    }

    float topPThreshold = topPs[row] * totalWeight;
    float selectedWeight = 0.0f;
    int selectedCount = topK;
    for (int index = 0; index < topK; ++index) {
      selectedWeight += samplingWeight(
          candidateLogits[candidateOffset + index], maxLogit, temperatures[row]);
      if (selectedWeight >= topPThreshold) {
        selectedCount = index + 1;
        break;
      }
    }

    float draw = uniform * selectedWeight;
    float cumulativeWeight = 0.0f;
    int sampledIndex = selectedCount - 1;
    for (int index = 0; index < selectedCount; ++index) {
      cumulativeWeight += samplingWeight(
          candidateLogits[candidateOffset + index], maxLogit, temperatures[row]);
      if (cumulativeWeight > draw) {
        sampledIndex = index;
        break;
      }
    }
    result[row] = candidateLabels[candidateOffset + sampledIndex];
  }
}

template<typename T>
Tensor sampleLogitsImpl(
    const Tensor &logits,
    const Tensor &uniformNoise,
    const Tensor &temperatures,
    const Tensor &topKs,
    const Tensor &topPs) {
  int rows = logits.getShape(0);
  int vocabSize = logits.getShape(1);
  int64_t candidateCount64 = static_cast<int64_t>(rows) * vocabSize;
  CHECK(candidateCount64 <= std::numeric_limits<int>::max());
  int candidateCount = static_cast<int>(candidateCount64);

  lut::c_ptr<uint32_t> thresholds = llynCudaAlloc<uint32_t>(rows);
  lut::c_ptr<float> candidateLogitsIn = llynCudaAlloc<float>(candidateCount);
  lut::c_ptr<float> candidateLogitsOut = llynCudaAlloc<float>(candidateCount);
  lut::c_ptr<int> candidateLabelsIn = llynCudaAlloc<int>(candidateCount);
  lut::c_ptr<int> candidateLabelsOut = llynCudaAlloc<int>(candidateCount);
  lut::c_ptr<int> beginOffsets = llynCudaAlloc<int>(rows);
  lut::c_ptr<int> endOffsets = llynCudaAlloc<int>(rows);

  radixSelectTopKThresholdKernel<T><<<rows, BlockSize>>>(
      getDataPtrCuda<T>(logits),
      getDataPtrCuda<float>(temperatures),
      getDataPtrCuda<IntType>(topKs),
      thresholds.get(),
      rows,
      vocabSize);
  gatherTopKCandidatesKernel<T><<<rows, BlockSize>>>(
      getDataPtrCuda<T>(logits),
      getDataPtrCuda<float>(temperatures),
      getDataPtrCuda<IntType>(topKs),
      thresholds.get(),
      candidateLogitsIn.get(),
      candidateLabelsIn.get(),
      rows,
      vocabSize);
  initializeCandidateOffsetsKernel<<<getGrid1D(rows, BlockSize), BlockSize>>>(
      getDataPtrCuda<float>(temperatures),
      getDataPtrCuda<IntType>(topKs),
      beginOffsets.get(),
      endOffsets.get(),
      rows,
      vocabSize);

  size_t tempStorageBytes = 0;
  LL_CHECK_CUDA_STATUS(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      tempStorageBytes,
      candidateLogitsIn.get(),
      candidateLogitsOut.get(),
      candidateLabelsIn.get(),
      candidateLabelsOut.get(),
      candidateCount,
      rows,
      beginOffsets.get(),
      endOffsets.get()));
  lut::c_ptr<std::byte> tempStorage = llynCudaAlloc<std::byte>(tempStorageBytes);
  LL_CHECK_CUDA_STATUS(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      tempStorage.get(),
      tempStorageBytes,
      candidateLogitsIn.get(),
      candidateLogitsOut.get(),
      candidateLabelsIn.get(),
      candidateLabelsOut.get(),
      candidateCount,
      rows,
      beginOffsets.get(),
      endOffsets.get()));

  Tensor result = createCudaTensorLong({rows});
  sampleTopPFromSortedCandidatesKernel<<<rows, BlockSize>>>(
      candidateLogitsOut.get(),
      candidateLabelsOut.get(),
      getDataPtrCuda<float>(uniformNoise),
      getDataPtrCuda<float>(temperatures),
      getDataPtrCuda<IntType>(topKs),
      getDataPtrCuda<float>(topPs),
      rows,
      vocabSize,
      getDataPtrCuda<LongType>(result));
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return result;
}

}  // namespace

Tensor sample(
    const Tensor &logits,
    const Tensor &uniformNoise,
    const Tensor &temperatures,
    const Tensor &topKs,
    const Tensor &topPs) {
  CHECK(logits.getDevice().getType() == Device::kCuda && logits.getDim() == 2 &&
        logits.isContiguous());
  int rows = logits.getShape(0);
  CHECK(uniformNoise.getDevice().getType() == Device::kCuda &&
        uniformNoise.getDType() == DType::kFloat && uniformNoise.isContiguous() &&
        uniformNoise.getShape() == std::vector<int>({rows}));
  CHECK(temperatures.getDevice().getType() == Device::kCuda &&
        temperatures.getDType() == DType::kFloat && temperatures.isContiguous() &&
        temperatures.getShape() == std::vector<int>({rows}));
  CHECK(topKs.getDevice().getType() == Device::kCuda &&
        topKs.getDType() == DType::kInt32 && topKs.isContiguous() &&
        topKs.getShape() == std::vector<int>({rows}));
  CHECK(topPs.getDevice().getType() == Device::kCuda &&
        topPs.getDType() == DType::kFloat && topPs.isContiguous() &&
        topPs.getShape() == std::vector<int>({rows}));

  if (logits.getDType() == DType::kFloat16) {
    return sampleLogitsImpl<half>(logits, uniformNoise, temperatures, topKs, topPs);
  }
  if (logits.getDType() == DType::kFloat) {
    return sampleLogitsImpl<float>(logits, uniformNoise, temperatures, topKs, topPs);
  }
  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
