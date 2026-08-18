#include "flint/cuda/sampling.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>

#include <cfloat>

#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {

template<typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void collectTopKCandidatesKernel(
    const T *__restrict__ distribution,
    float *__restrict__ probabilities,
    int *__restrict__ labels,
    int rows,
    int vocabSize,
    int blocksPerRow,
    int topK) {
  int row = blockIdx.x / blocksPerRow;
  int rowBlock = blockIdx.x % blocksPerRow;
  if (row >= rows) return;

  float threadProbabilities[ITEMS_PER_THREAD];
  int threadLabels[ITEMS_PER_THREAD];
  int blockOffset = rowBlock * BLOCK_SIZE * ITEMS_PER_THREAD;

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int label = blockOffset + threadIdx.x * ITEMS_PER_THREAD + item;
    threadProbabilities[item] =
      label < vocabSize
      ? static_cast<float>(distribution[row * vocabSize + label])
      : -INFINITY;
    threadLabels[item] = label;
  }

  using BlockSort = cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockSort::TempStorage tempStorage;
  BlockSort(tempStorage).SortDescending(threadProbabilities, threadLabels);

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int rank = threadIdx.x * ITEMS_PER_THREAD + item;
    if (rank < topK) {
      int outputIndex = (row * blocksPerRow + rowBlock) * topK + rank;
      probabilities[outputIndex] = threadProbabilities[item];
      labels[outputIndex] = threadLabels[item];
    }
  }
}

template<>
__global__ void collectTopKCandidatesKernel<half, 256, 4>(
    const half *__restrict__ distribution,
    float *__restrict__ probabilities,
    int *__restrict__ labels,
    int rows,
    int vocabSize,
    int blocksPerRow,
    int topK) {
  constexpr int blockSize = 256;
  constexpr int itemsPerThread = 4;
  float threadProbabilities[itemsPerThread];
  int threadLabels[itemsPerThread];
  int row = blockIdx.x / blocksPerRow;
  int rowBlock = blockIdx.x % blocksPerRow;
  if (row >= rows) return;
  int blockOffset = rowBlock * blockSize * itemsPerThread;

#pragma unroll
  for (int item = 0; item < itemsPerThread; ++item) {
    int label = blockOffset + threadIdx.x * itemsPerThread + item;
    threadProbabilities[item] =
      label < vocabSize
      ? __half2float(distribution[row * vocabSize + label])
      : -INFINITY;
    threadLabels[item] = label;
  }

  using BlockSort = cub::BlockRadixSort<float, blockSize, itemsPerThread, int>;
  __shared__ typename BlockSort::TempStorage tempStorage;
  BlockSort(tempStorage).SortDescending(threadProbabilities, threadLabels);

#pragma unroll
  for (int item = 0; item < itemsPerThread; ++item) {
    int rank = threadIdx.x * itemsPerThread + item;
    if (rank < topK) {
      int outputIndex = (row * blocksPerRow + rowBlock) * topK + rank;
      probabilities[outputIndex] = threadProbabilities[item];
      labels[outputIndex] = threadLabels[item];
    }
  }
}

template<typename T>
__global__ void initializeCandidatesKernel(
    const T *__restrict__ distribution,
    float *__restrict__ probabilities,
    int *__restrict__ labels,
    int numel,
    int vocabSize) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numel) return;

  probabilities[index] = static_cast<float>(distribution[index]);
  labels[index] = index % vocabSize;
}

__global__ void initializeSegmentOffsetsKernel(
    int *__restrict__ offsets,
    int rows,
    int candidatesPerRow) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index <= rows) offsets[index] = index * candidatesPerRow;
}

template<int BLOCK_SIZE>
__global__ void sampleTopPExponentialRaceKernel(
    const float *__restrict__ probabilities,
    const int *__restrict__ labels,
    const float *__restrict__ uniformNoise,
    int rows,
    int candidatesPerRow,
    int topK,
    float topP,
    LongType *__restrict__ result) {
  int row = blockIdx.x;
  if (row >= rows) return;

  __shared__ int selectedCount;
  if (threadIdx.x == 0) {
    float selectedSum = 0.0f;
    selectedCount = 0;
    for (int index = 0; index < topK; ++index) {
      selectedSum += probabilities[row * candidatesPerRow + index];
      ++selectedCount;
      if (selectedSum >= topP) break;
    }
  }
  __syncthreads();

  cub::KeyValuePair<int, float> threadBest(0, -INFINITY);
  for (int index = threadIdx.x; index < selectedCount; index += BLOCK_SIZE) {
    float probability = probabilities[row * candidatesPerRow + index];
    float exponentialNoise = fmaxf(-logf(uniformNoise[row * topK + index]), FLT_MIN);
    float score = probability / exponentialNoise;
    if (score > threadBest.value) {
      threadBest = cub::KeyValuePair<int, float>(index, score);
    }
  }

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  cub::KeyValuePair<int, float> blockBest =
      BlockReduce(tempStorage).Reduce(threadBest, cub::ArgMax());
  if (threadIdx.x == 0) {
    result[row] = labels[row * candidatesPerRow + blockBest.key];
  }
}

template<typename T>
Tensor sampleImpl(
    const Tensor &distribution,
    const Tensor &uniformNoise,
    int topK,
    float topP) {
  int64_t numel64 = distribution.getNumEl();
  CHECK(numel64 <= std::numeric_limits<int>::max());
  int numel = static_cast<int>(numel64);
  int vocabSize = distribution.getShape(-1);
  int rows = numel / vocabSize;
  CHECK(topK > 0 && topK <= vocabSize);
  CHECK(uniformNoise.getDim() == 2 && uniformNoise.getShape(0) == rows &&
        uniformNoise.getShape(1) == topK);

  constexpr int blockSize = 256;
  constexpr int itemsPerThread = 4;
  constexpr int itemsPerBlock = blockSize * itemsPerThread;
  int blocksPerRow = (vocabSize + itemsPerBlock - 1) / itemsPerBlock;
  bool useBlockTopK = topK <= itemsPerBlock;
  int candidatesPerRow = useBlockTopK ? blocksPerRow * topK : vocabSize;
  int64_t candidateCount64 = static_cast<int64_t>(rows) * candidatesPerRow;
  CHECK(candidateCount64 <= std::numeric_limits<int>::max());
  int candidateCount = static_cast<int>(candidateCount64);

  lut::c_ptr<float> probabilitiesIn = llynCudaAlloc<float>(candidateCount);
  lut::c_ptr<float> probabilitiesOut = llynCudaAlloc<float>(candidateCount);
  lut::c_ptr<int> labelsIn = llynCudaAlloc<int>(candidateCount);
  lut::c_ptr<int> labelsOut = llynCudaAlloc<int>(candidateCount);
  lut::c_ptr<int> segmentOffsets = llynCudaAlloc<int>(rows + 1);

  if (useBlockTopK) {
    int numBlocks = rows * blocksPerRow;
    collectTopKCandidatesKernel<T, blockSize, itemsPerThread><<<numBlocks, blockSize>>>(
        getDataPtrCuda<T>(distribution),
        probabilitiesIn.get(),
        labelsIn.get(),
        rows,
        vocabSize,
        blocksPerRow,
        topK);
  } else {
    dim3 grid = getGrid1D(numel, blockSize);
    initializeCandidatesKernel<<<grid, blockSize>>>(
        getDataPtrCuda<T>(distribution),
        probabilitiesIn.get(),
        labelsIn.get(),
        numel,
        vocabSize);
  }

  dim3 offsetGrid = getGrid1D(rows + 1, blockSize);
  initializeSegmentOffsetsKernel<<<offsetGrid, blockSize>>>(
      segmentOffsets.get(), rows, candidatesPerRow);

  void *tempStorage = nullptr;
  size_t tempStorageBytes = 0;
  LL_CHECK_CUDA_STATUS(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      tempStorage,
      tempStorageBytes,
      probabilitiesIn.get(),
      probabilitiesOut.get(),
      labelsIn.get(),
      labelsOut.get(),
      candidateCount,
      rows,
      segmentOffsets.get(),
      segmentOffsets.get() + 1));
  lut::c_ptr<std::byte> tempStoragePtr = llynCudaAlloc<std::byte>(tempStorageBytes);
  LL_CHECK_CUDA_STATUS(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      tempStoragePtr.get(),
      tempStorageBytes,
      probabilitiesIn.get(),
      probabilitiesOut.get(),
      labelsIn.get(),
      labelsOut.get(),
      candidateCount,
      rows,
      segmentOffsets.get(),
      segmentOffsets.get() + 1));

  std::vector<int> outputShape = distribution.getShape();
  outputShape.pop_back();
  if (outputShape.empty()) outputShape.push_back(1);
  Tensor result = createCudaTensorLong(outputShape);
  sampleTopPExponentialRaceKernel<blockSize><<<rows, blockSize>>>(
      probabilitiesOut.get(),
      labelsOut.get(),
      getDataPtrCuda<float>(uniformNoise),
      rows,
      candidatesPerRow,
      topK,
      topP,
      getDataPtrCuda<LongType>(result));
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return result;
}

Tensor sample(
    const Tensor &distribution,
    const Tensor &uniformNoise,
    int topK,
    float topP) {
  CHECK(distribution.getDevice().getType() == Device::kCuda);
  CHECK(distribution.getDim() >= 1);
  CHECK(distribution.isContiguous());
  CHECK(uniformNoise.getDevice().getType() == Device::kCuda &&
        uniformNoise.getDType() == DType::kFloat && uniformNoise.isContiguous());

  if (distribution.getDType() == DType::kFloat16) {
    return sampleImpl<half>(distribution, uniformNoise, topK, topP);
  }
  if (distribution.getDType() == DType::kFloat) {
    return sampleImpl<float>(distribution, uniformNoise, topK, topP);
  }
  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl