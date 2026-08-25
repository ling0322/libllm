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

#include <cuda_fp16.h>

#include "flint/cuda/causal_conv1d.h"
#include "flint/cuda/common.h"
#include "flint/cuda/to_device.h"

#include <algorithm>
#include <vector>

#ifdef LIBLLM_CAUSAL_CONV1D_ENABLED
#include "causal_conv1d_api.h"
#endif

namespace fl {
namespace op {
namespace cuda {

namespace {

constexpr int MaxKernelSize = 8;

/// One thread per (token, channel). The taps are read straight from global memory: the kernel is
/// short enough -- four for the models that use this -- that a shared-memory staging pass costs
/// more than the reads it saves.
template<typename T>
__global__ void causalConv1dKernel(
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const IntType *__restrict__ cuSeqlens,
    T *__restrict__ output,
    int numChannels,
    int kernelSize,
    int numSequences) {
  int sequence = blockIdx.y;
  if (sequence >= numSequences) return;

  int begin = cuSeqlens[sequence];
  int end = cuSeqlens[sequence + 1];
  int length = end - begin;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int token = index / numChannels;
  int channel = index - token * numChannels;
  if (token >= length) return;

  int position = begin + token;
  float sum = 0.0f;
#pragma unroll
  for (int tap = 0; tap < MaxKernelSize; ++tap) {
    if (tap >= kernelSize) break;
    // The window ends at `position`; reaching back past the start of this sequence reads as zero
    // rather than borrowing the tail of whatever was packed before it.
    int source = position - (kernelSize - 1) + tap;
    if (source < begin) continue;
    sum += static_cast<float>(weight[channel * kernelSize + tap]) *
           static_cast<float>(input[source * numChannels + channel]);
  }

  output[position * numChannels + channel] = static_cast<T>(sum);
}

template<typename T>
Tensor causalConv1dImpl(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &cuSeqlens,
    int numSequences,
    int maxLength) {
  int numTokens = input.getShape(0);
  int numChannels = input.getShape(1);
  int kernelSize = weight.getShape(1);

  Tensor output = createCudaTensor<T>({numTokens, numChannels});
  if (numSequences == 0 || maxLength == 0) return output;

  constexpr int blockSize = 256;
  dim3 grid;
  grid.x = (maxLength * numChannels + blockSize - 1) / blockSize;
  grid.y = numSequences;

  causalConv1dKernel<T><<<grid, blockSize>>>(
      getDataPtrCuda<T>(input),
      getDataPtrCuda<T>(weight),
      getDataPtrCuda<IntType>(cuSeqlens),
      getDataPtrCuda<T>(output),
      numChannels,
      kernelSize,
      numSequences);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return output;
}

}  // namespace

Tensor causalConv1dFallback(const Tensor &input, const Tensor &weight, const Tensor &cuSeqlens) {
  CHECK(input.getDim() == 2 && weight.getDim() == 2 && cuSeqlens.getDim() == 1);
  CHECK(input.getDevice().getType() == Device::kCuda);
  CHECK(weight.getDevice().getType() == Device::kCuda);
  CHECK(cuSeqlens.getDevice().getType() == Device::kCuda);
  CHECK(cuSeqlens.getDType() == DType::kInt32);
  LL_CHECK_CONTIGUOUS(input);
  LL_CHECK_CONTIGUOUS(weight);
  LL_CHECK_CONTIGUOUS(cuSeqlens);

  int numChannels = input.getShape(1);
  int kernelSize = weight.getShape(1);
  CHECK(weight.getShape(0) == numChannels);
  CHECK(kernelSize > 0 && kernelSize <= MaxKernelSize);

  int numSequences = cuSeqlens.getShape(0) - 1;
  CHECK(numSequences >= 0);

  // The grid is sized by the longest sequence, so the boundaries have to be read on the host.
  Tensor hostOffsets = toCpu(cuSeqlens);
  const IntType *offsets = hostOffsets.getInternalData()->getData<IntType>(
      hostOffsets.getInternalOffset());
  int maxLength = 0;
  for (int i = 0; i < numSequences; ++i) {
    CHECK(offsets[i] >= 0 && offsets[i] <= offsets[i + 1]);
    CHECK(offsets[i + 1] <= input.getShape(0));
    maxLength = std::max(maxLength, offsets[i + 1] - offsets[i]);
  }

  if (input.getDType() == DType::kFloat16)
    return causalConv1dImpl<half>(input, weight, cuSeqlens, numSequences, maxLength);
  if (input.getDType() == DType::kFloat)
    return causalConv1dImpl<float>(input, weight, cuSeqlens, numSequences, maxLength);

  NOT_IMPL();
}

#ifdef LIBLLM_CAUSAL_CONV1D_ENABLED

namespace {

/// Labels every packed position with the index of the sequence it belongs to. That is how the
/// vendored kernel is told where a sequence ends: it only accumulates taps whose label matches
/// the position being written, so a window never reaches into its neighbour.
Tensor buildSeqIdx(const Tensor &cuSeqlens, int numTokens, int numSequences) {
  Tensor hostOffsets = toCpu(cuSeqlens);
  const IntType *offsets = hostOffsets.getInternalData()->getData<IntType>(
      hostOffsets.getInternalOffset());

  std::vector<IntType> labels(numTokens, 0);
  for (int sequence = 0; sequence < numSequences; ++sequence) {
    for (int token = offsets[sequence]; token < offsets[sequence + 1]; ++token) {
      labels[token] = sequence;
    }
  }

  return toCuda(Tensor::create<IntType>({numTokens}, labels));
}

}  // namespace

/// Runs the vendored kernel over the whole packed batch as one "sequence" of numTokens, relying
/// on seq_idx for the boundaries. The kernel wants a channel-last layout, which is exactly how a
/// packed batch is already stored.
Tensor causalConv1dVendored(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &cuSeqlens,
    int numSequences) {
  int numTokens = input.getShape(0);
  int numChannels = input.getShape(1);
  int kernelSize = weight.getShape(1);
  bool isHalf = input.getDType() == DType::kFloat16;

  Tensor output = isHalf ? createCudaTensorHalf({numTokens, numChannels})
                         : createCudaTensorFloat({numTokens, numChannels});
  if (numTokens == 0) return output;

  Tensor seqIdx = buildSeqIdx(cuSeqlens, numTokens, numSequences);

  ConvParamsBase params = {};
  params.batch = 1;
  params.dim = numChannels;
  params.seqlen = numTokens;
  params.width = kernelSize;
  params.silu_activation = false;

  params.x_ptr = const_cast<void *>(static_cast<const void *>(getDataPtrCuda<void>(input)));
  params.x_batch_stride = static_cast<uint32_t>(numTokens) * numChannels;
  params.x_c_stride = 1;
  params.x_l_stride = numChannels;

  params.weight_ptr = const_cast<void *>(static_cast<const void *>(getDataPtrCuda<void>(weight)));
  params.weight_c_stride = kernelSize;
  params.weight_width_stride = 1;

  params.out_ptr = getDataPtrCuda<void>(output);
  params.out_batch_stride = static_cast<uint32_t>(numTokens) * numChannels;
  params.out_c_stride = 1;
  params.out_l_stride = numChannels;

  params.bias_ptr = nullptr;
  params.seq_idx_ptr = getDataPtrCuda<void>(seqIdx);

  causal_conv1d::channellast_fwd(params, isHalf, /*stream=*/0);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return output;
}

#endif  // LIBLLM_CAUSAL_CONV1D_ENABLED

Tensor causalConv1d(const Tensor &input, const Tensor &weight, const Tensor &cuSeqlens) {
  CHECK(input.getDim() == 2 && weight.getDim() == 2 && cuSeqlens.getDim() == 1);
  CHECK(cuSeqlens.getDType() == DType::kInt32);
  LL_CHECK_CONTIGUOUS(input);
  LL_CHECK_CONTIGUOUS(weight);
  CHECK(weight.getShape(0) == input.getShape(1));

  int numSequences = cuSeqlens.getShape(0) - 1;
  CHECK(numSequences >= 0);

#ifdef LIBLLM_CAUSAL_CONV1D_ENABLED
  // The vendored kernel handles widths 2 to 4, which is what the models that use it need. A width
  // of 1 is a per-channel scale and is left to the portable path.
  int kernelSize = weight.getShape(1);
  if (kernelSize >= 2 && kernelSize <= 4 && numSequences > 0) {
    return causalConv1dVendored(input, weight, cuSeqlens, numSequences);
  }
#endif

  return causalConv1dFallback(input, weight, cuSeqlens);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
