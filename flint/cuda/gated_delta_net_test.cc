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

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/cuda/gated_delta_net.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace {

class Lcg {
 public:
  explicit Lcg(uint64_t seed)
      : _state(seed) {
  }

  // A value in [-0.5, 0.5).
  float next() {
    _state = _state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((_state >> 40) & 0xffff) / 65536.0f - 0.5f;
  }

 private:
  uint64_t _state;
};

void normalizeRows(std::vector<float> &data, int rows, int dim) {
  for (int r = 0; r < rows; ++r) {
    float *row = data.data() + static_cast<size_t>(r) * dim;
    float sum = 0.0f;
    for (int c = 0; c < dim; ++c) sum += row[c] * row[c];

    float scale = 1.0f / std::sqrt(sum + 1e-6f);
    for (int c = 0; c < dim; ++c) row[c] *= scale;
  }
}

Tensor toCudaHalf(const Tensor &a) {
  return F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCudaFloat(const Tensor &a) {
  return F::to(Device::getCuda(), a);
}

Tensor toCpuFloat(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

// The regimes the decays and the write strengths are drawn from. The default is what a trained
// model looks like; the others are the ends the kernels have to survive rather than the middle.
enum class Regime {
  kDefault,
  // No decay at all. Every exp(c_i - c_j) is one, so nothing that divides or subtracts logs may
  // drift, and the chunk's system is the plain delta rule.
  kNoDecay,
  // Decay strong enough that exp(c) underflows within a chunk: at -6 a token, exp(c_63) is about
  // e-164, which is zero in float long before the chunk ends. Both tensor core paths scale queries
  // or the output by that number, and what they must not do is turn the tokens whose own decay is
  // still O(1) into zero along with it.
  kStrongDecay,
  // Beta at the ends of the range the delta rule allows: zero writes nothing, two overshoots.
  kBetaEnds,
};

// How the state slots are handed out. Reversed is the usual case; scattered adds a pool that is
// twice the size it needs to be and a mapping that is neither monotonic nor contiguous.
enum class SlotPolicy {
  kReversed,
  kScattered,
};

// Run the same prefill on both backends and compare the output and the state they leave behind.
// The CUDA kernels carry q, k, v and the output in half, so the tolerance is a half precision one;
// the state, the decays and the triangular systems stay in float on both sides.
void compareBackends(
    const std::vector<int> &seqlens,
    int numKHead,
    int headRatio,
    int headDim,
    bool nonZeroState,
    uint64_t seed,
    float tolerance,
    op::cuda::GatedDeltaNetPath path,
    Regime regime = Regime::kDefault,
    SlotPolicy slotPolicy = SlotPolicy::kReversed) {
  Lcg lcg(seed);
  int numVHead = numKHead * headRatio;
  int numSeq = static_cast<int>(seqlens.size());

  std::vector<int32_t> cuSeqlens;
  cuSeqlens.push_back(0);
  for (int len : seqlens) cuSeqlens.push_back(cuSeqlens.back() + len);
  int numTokens = cuSeqlens.back();

  std::vector<float> qData(static_cast<size_t>(numTokens) * numKHead * headDim);
  std::vector<float> kData(qData.size());
  std::vector<float> vData(static_cast<size_t>(numTokens) * numVHead * headDim);
  std::vector<float> gData(static_cast<size_t>(numTokens) * numVHead);
  std::vector<float> betaData(gData.size());
  // A pool with a slot to spare, in which sequence i owns slot numSeq - i. Both backends are
  // handed the same mapping, so a kernel that reached the state by position in the batch instead
  // would disagree with the reference on every sequence but the middle one, and would leave the
  // spare slot's sentinel overwritten.
  const int numStateSlot = (slotPolicy == SlotPolicy::kScattered) ? 2 * numSeq + 1 : numSeq + 1;
  std::vector<int32_t> stateSlots;
  for (int seq = 0; seq < numSeq; ++seq) {
    if (slotPolicy == SlotPolicy::kScattered) {
      // Neither monotonic nor contiguous, and never slot 0, which stays a sentinel.
      stateSlots.push_back(1 + (seq * 7 + 3) % (2 * numSeq));
    } else {
      stateSlots.push_back(numSeq - seq);
    }
  }

  std::vector<float> stateData(
      static_cast<size_t>(numStateSlot) * numVHead * headDim * headDim, 7.0f);
  const size_t slotSize = static_cast<size_t>(numVHead) * headDim * headDim;
  std::fill(stateData.begin() + slotSize, stateData.end(), 0.0f);

  for (float &x : qData) x = lcg.next();
  for (float &x : kData) x = lcg.next();
  for (float &x : vData) x = lcg.next();
  normalizeRows(qData, numTokens * numKHead, headDim);
  normalizeRows(kData, numTokens * numKHead, headDim);
  switch (regime) {
    case Regime::kNoDecay:
      for (float &x : gData) x = 0.0f;
      for (float &x : betaData) x = lcg.next() + 0.5f;
      break;
    case Regime::kStrongDecay:
      for (float &x : gData) x = -(lcg.next() + 0.5f) * 6.0f;
      for (float &x : betaData) x = lcg.next() + 0.5f;
      break;
    case Regime::kBetaEnds:
      for (float &x : gData) x = -(lcg.next() + 0.5f) * 0.5f;
      for (size_t i = 0; i < betaData.size(); ++i) betaData[i] = (i % 2 == 0) ? 0.0f : 2.0f;
      break;
    default:
      for (float &x : gData) x = -(lcg.next() + 0.5f) * 0.5f;
      for (float &x : betaData) x = lcg.next() + 0.5f;
      break;
  }
  if (nonZeroState) {
    for (size_t i = slotSize; i < stateData.size(); ++i) stateData[i] = lcg.next() * 0.2f;
  }

  Tensor q = Tensor::create<float>({numTokens, numKHead, headDim}, lut::makeConstSpan(qData));
  Tensor k = Tensor::create<float>({numTokens, numKHead, headDim}, lut::makeConstSpan(kData));
  Tensor v = Tensor::create<float>({numTokens, numVHead, headDim}, lut::makeConstSpan(vData));
  Tensor g = Tensor::create<float>({numTokens, numVHead}, lut::makeConstSpan(gData));
  Tensor beta = Tensor::create<float>({numTokens, numVHead}, lut::makeConstSpan(betaData));
  Tensor seqlensTensor = Tensor::create<int32_t>(
      {numSeq + 1},
      lut::makeConstSpan(cuSeqlens));
  Tensor slotsTensor = Tensor::create<int32_t>({numSeq}, lut::makeConstSpan(stateSlots));

  // Both backends overwrite the state, so each gets its own copy of the pool.
  Tensor stateCpu = Tensor::create<float>(
      {numStateSlot, numVHead, headDim, headDim},
      lut::makeConstSpan(stateData));
  Tensor stateCuda = toCudaFloat(Tensor::create<float>(
      {numStateSlot, numVHead, headDim, headDim},
      lut::makeConstSpan(stateData)));

  // The path is forced rather than left to kAuto: these head counts are far below what kAuto reads
  // as enough CTAs to fuse, so kAuto alone would never reach the fused kernel here.
  Tensor expected = F::gatedDeltaNetPrefill(
      q,
      k,
      v,
      g,
      beta,
      seqlensTensor,
      slotsTensor,
      stateCpu);
  Tensor actual = op::cuda::gatedDeltaNetPrefill(
      toCudaHalf(q),
      toCudaHalf(k),
      toCudaHalf(v),
      toCudaFloat(g),
      toCudaFloat(beta),
      toCudaFloat(seqlensTensor),
      toCudaFloat(slotsTensor),
      stateCuda,
      path);

  CATCH_REQUIRE(actual.getShape() == std::vector<int>{numTokens, numVHead, headDim});
  CATCH_REQUIRE(F::allClose(toCpuFloat(actual), expected, tolerance, tolerance));
  CATCH_REQUIRE(F::allClose(toCpuFloat(stateCuda), stateCpu, tolerance, tolerance));
}

constexpr op::cuda::GatedDeltaNetPath kPaths[] = {
    op::cuda::GatedDeltaNetPath::kTensorCoreMma,
    op::cuda::GatedDeltaNetPath::kTensorCoreMmaChunkOnly,
};

}  // namespace

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Inside one chunk, exactly one chunk, one chunk plus a token, and several chunks. The two paths
  // factorise the sequence at different chunk lengths, 32 and 64, so a length that is exact for one
  // is partial for the other.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    for (int len : {1, 31, 32, 33, 63, 64, 65, 200}) {
      CATCH_INFO("len = " << len);
      compareBackends({len}, 1, 1, 32, false, 0x1000 + len, 5e-3f, path);
    }
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (incoming state)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A non-zero incoming state is the only thing that puts the state into the right hand side.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    compareBackends({150}, 2, 1, 64, true, 0x2000, 5e-3f, path);
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (varlen batch)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    compareBackends({5, 64, 100, 1, 0}, 2, 1, 32, true, 0x3000, 5e-3f, path);
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (grouped heads)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Three value heads per key head at a head dimension of 128, the Qwen3.5 shape.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    compareBackends({70, 33}, 2, 3, 128, true, 0x4000, 5e-3f, path);
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (tile edges)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Lengths around the edges of the 16 by 8 tiles the accumulators are made of, which is where a
  // mask that is off by a row or a column shows up. 8 and 16 are exact tiles, 9 and 17 leave one
  // token in the next one, and 48 fills three of the four row blocks of a chunk exactly.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    for (int len : {7, 8, 9, 15, 16, 17, 47, 48, 49}) {
      CATCH_INFO("len = " << len);
      compareBackends({len}, 1, 1, 64, true, 0x6000 + len, 5e-3f, path);
    }
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (decode batch)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // What a decode step is: every sequence one token, on top of a state it has to carry. The mma
  // path steps these rather than chunking them, and this is the only case that runs that at the
  // shape the model does -- three value heads per key head, a head dimension of 128.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    compareBackends({1, 1, 1, 1, 1, 1, 1, 1}, 2, 3, 128, true, 0x7200, 5e-3f, path);
  }

  // And a mixed step: two sequences still prefilling next to six that are decoding, with lengths
  // either side of the crossover. That batch is what the branch inside the kernel exists for, and
  // both sides of it have to leave the same state behind for the next one.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    compareBackends({1, 1, 200, 16, 1, 17, 1, 1}, 2, 3, 128, true, 0x7300, 5e-3f, path);
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (many chunks)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Five chunks and a one-token tail, at the shape the model actually runs: the state has to be
  // carried from one chunk to the next four times over, and the last chunk is as short as it gets.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    compareBackends({257}, 2, 3, 128, true, 0x7000, 5e-3f, path);
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (decay regimes)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    // No decay: every exp(c_i - c_j) is exactly one.
    compareBackends({130}, 1, 2, 64, true, 0x7100, 5e-3f, path, Regime::kNoDecay);

    // Decay strong enough to underflow inside a chunk. The tolerance is absolute as well as
    // relative, and everything the decay kills is legitimately zero, so this is not looser.
    compareBackends({130}, 1, 2, 64, true, 0x7200, 5e-3f, path, Regime::kStrongDecay);

    // Beta at both ends: half the tokens write nothing, the other half overshoot.
    compareBackends({130}, 1, 2, 64, true, 0x7300, 5e-3f, path, Regime::kBetaEnds);
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (scattered slots)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A pool with twice the slots the batch needs and a mapping that is neither monotonic nor
  // contiguous, over a batch that also has empty sequences at both ends. An empty sequence must
  // leave its slot exactly as it found it, which the whole-pool comparison checks.
  for (op::cuda::GatedDeltaNetPath path : kPaths) {
    compareBackends(
        {0, 70, 1, 0, 65, 0},
        2,
        1,
        64,
        true,
        0x7400,
        5e-3f,
        path,
        Regime::kDefault,
        SlotPolicy::kScattered);
  }
}

CATCH_TEST_CASE("test CUDA gatedDeltaNetPrefill (auto)", "[op][cuda][gated_delta_net]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // kAuto is kTensorCoreMma now, but it is what a caller reaches for, so it is run at every head
  // dimension the kernel is instantiated for rather than trusted to be the same code.
  for (int headDim : {32, 64, 128}) {
    CATCH_INFO("headDim = " << headDim);
    compareBackends(
        {130, 0, 7},
        1,
        2,
        headDim,
        true,
        0x8000 + headDim,
        5e-3f,
        op::cuda::GatedDeltaNetPath::kAuto);
  }
}

}  // namespace fl
