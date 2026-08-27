// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "flint/cuda/gated_delta_net.h"

#include <cuda_fp16.h>

#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {

bool tensorCoreAvailable() {
  // Asked on every layer of every forward pass, and the answer cannot change under a process.
  static const bool available = [] {
    int major = 0;
    int minor = 0;
    LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    return major * 10 + minor >= gdnmma::kMinArch;
  }();
  return available;
}

Tensor gatedDeltaNetPrefill(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &g,
    const Tensor &beta,
    const Tensor &cuSeqlens,
    const Tensor &stateSlots,
    Tensor &state,
    GatedDeltaNetPath path) {
  CHECK(q.getDevice().getType() == Device::kCuda);
  CHECK(k.getDevice().getType() == Device::kCuda);
  CHECK(v.getDevice().getType() == Device::kCuda);
  CHECK(g.getDevice().getType() == Device::kCuda);
  CHECK(beta.getDevice().getType() == Device::kCuda);
  CHECK(cuSeqlens.getDevice().getType() == Device::kCuda);
  CHECK(stateSlots.getDevice().getType() == Device::kCuda);
  CHECK(state.getDevice().getType() == Device::kCuda);
  CHECK(q.getDType() == DType::kFloat16 && k.getDType() == DType::kFloat16);
  CHECK(v.getDType() == DType::kFloat16);
  CHECK(g.getDType() == DType::kFloat && beta.getDType() == DType::kFloat);
  CHECK(state.getDType() == DType::kFloat);
  CHECK(cuSeqlens.getDType() == DType::kInt32 && stateSlots.getDType() == DType::kInt32);
  LL_CHECK_CONTIGUOUS(q);
  LL_CHECK_CONTIGUOUS(k);
  LL_CHECK_CONTIGUOUS(v);
  LL_CHECK_CONTIGUOUS(g);
  LL_CHECK_CONTIGUOUS(beta);
  LL_CHECK_CONTIGUOUS(cuSeqlens);
  LL_CHECK_CONTIGUOUS(stateSlots);
  LL_CHECK_CONTIGUOUS(state);
  CHECK(q.getDim() == 3 && k.getDim() == 3 && v.getDim() == 3);
  CHECK(g.getDim() == 2 && beta.getDim() == 2 && cuSeqlens.getDim() == 1);
  CHECK(stateSlots.getDim() == 1);
  CHECK(state.getDim() == 4);

  int numTokens = q.getShape(0);
  int numKHead = q.getShape(1);
  int headDim = q.getShape(2);
  int numVHead = v.getShape(1);
  int numSeq = cuSeqlens.getShape(0) - 1;

  CHECK(k.getShape(0) == numTokens && k.getShape(1) == numKHead && k.getShape(2) == headDim);
  CHECK(v.getShape(0) == numTokens && v.getShape(2) == headDim);
  CHECK(g.getShape(0) == numTokens && g.getShape(1) == numVHead);
  CHECK(beta.getShape(0) == numTokens && beta.getShape(1) == numVHead);
  CHECK(numVHead % numKHead == 0) << "the value heads must be a multiple of the key heads";
  CHECK(stateSlots.getShape(0) == numSeq) << "one state slot per sequence";
  CHECK(state.getShape(0) >= numSeq && state.getShape(1) == numVHead);
  CHECK(state.getShape(2) == headDim && state.getShape(3) == headDim);

  // There is one implementation now, so what it takes is what this operator takes, and the whole of
  // it is decided here. `gdnmma::fits` answers for the device and the head dimension together --
  // compute capability, the head dimensions the kernel is instantiated for, and whether the shared
  // memory it wants can be opted into -- so the failure is one message rather than something
  // discovered inside a fallback that no longer exists.
  int smem = 0;
  CHECK(gdnmma::fits(headDim, &smem))
      << "gatedDeltaNetPrefill needs a head dimension of 32, 64 or 128 and compute capability "
      << gdnmma::kMinArch / 10 << "." << gdnmma::kMinArch % 10 << " or later; this is head "
      << "dimension " << headDim << " on a device that "
      << (tensorCoreAvailable() ? "is new enough" : "is not");

  Tensor o = createCudaTensorHalf({numTokens, numVHead, headDim});
  if (numSeq == 0 || numTokens == 0) return o;

  // kAuto and kTensorCoreMma are the same thing now that there is one implementation; what is left
  // of `path` is whether the short sequences get stepped or chunked.
  int recurrentMax = path == GatedDeltaNetPath::kTensorCoreMmaChunkOnly
                         ? 0
                         : gdnmma::kDefaultRecurrentLen;
  gdnmma::run(
      q,
      k,
      v,
      g,
      beta,
      cuSeqlens,
      stateSlots,
      state,
      o,
      numKHead,
      numVHead,
      headDim,
      numSeq,
      recurrentMax);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return o;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
