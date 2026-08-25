// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Exercises CudaOperators::attention (cuda_operators.cc), which uses FlashAttention when it is
// compiled in and falls back to the portable operator otherwise -- unlike flash_attn_test.cc,
// this file is built whenever WITH_CUDA is on so both paths stay covered.

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toCuda(const Tensor &a) {
  return F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

}  // namespace

CATCH_TEST_CASE("test CUDA attention", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](int numHeads,
                    int numKeyValueHeads,
                    int queryLength,
                    int keyValueLength,
                    int headDim,
                    bool causal) {
    // a model feeds attention with [batch, length, heads, headDim] transposed to
    // [batch, heads, length, headDim], so the inputs are not contiguous.
    Tensor q = F::rand({1, queryLength, numHeads, headDim}, DType::kFloat);
    Tensor k = F::rand({1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
    Tensor v = F::rand({1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
    Tensor xr = F::attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal);

    Tensor x = F::attention(
        toCuda(q).transpose(1, 2),
        toCuda(k).transpose(1, 2),
        toCuda(v).transpose(1, 2),
        causal);

    return F::allClose(toCpu(x), xr, 5e-3f);
  };

  // headDim 128 goes to FlashAttention.
  CATCH_REQUIRE(runCase(8, 8, 8, 8, 128, false));
  CATCH_REQUIRE(runCase(8, 8, 128, 128, 128, false));
  CATCH_REQUIRE(runCase(8, 8, 8, 8, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 8, 8, 128, false));
  CATCH_REQUIRE(runCase(8, 2, 6, 10, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 1, 10, 128, true));

  // long enough to make the operator split the keys.
  CATCH_REQUIRE(runCase(8, 2, 1, 2048, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 4, 1024, 128, true));

  // headDim 16 is unsupported by FlashAttention and falls back to the portable path.
  CATCH_REQUIRE(runCase(8, 8, 16, 16, 16, false));
  CATCH_REQUIRE(runCase(8, 2, 6, 10, 16, true));
}

namespace {

bool runAttentionCase(
    int numHeads,
    int numKeyValueHeads,
    int queryLength,
    int keyValueLength,
    int headDim,
    bool causal) {
  Tensor q = F::rand({1, queryLength, numHeads, headDim}, DType::kFloat);
  Tensor k = F::rand({1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor v = F::rand({1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor xr = F::attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal);

  Tensor x = F::attention(
      toCuda(q).transpose(1, 2),
      toCuda(k).transpose(1, 2),
      toCuda(v).transpose(1, 2),
      causal);

  return F::allClose(toCpu(x), xr, 5e-3f);
}

}  // namespace

CATCH_TEST_CASE("test CUDA attention (head dims)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // FlashAttention only has kernels for a few head dims; the rest fall back to the portable
  // operator. Both sides of that split have to agree with the CPU reference.
  for (bool causal : {false, true}) {
    for (int headDim : {16, 32, 64, 128, 256}) {
      CATCH_INFO("headDim = " << headDim << ", causal = " << causal);
      CATCH_REQUIRE(runAttentionCase(4, 4, 8, 8, headDim, causal));
      CATCH_REQUIRE(runAttentionCase(4, 2, 6, 10, headDim, causal));
    }
  }
}

CATCH_TEST_CASE("test CUDA attention (degenerate lengths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single query against a single key is the shortest possible attention, and with one head it
  // removes the head axis from the indexing arithmetic entirely.
  for (bool causal : {false, true}) {
    CATCH_INFO("causal = " << causal);
    CATCH_REQUIRE(runAttentionCase(1, 1, 1, 1, 128, causal));
    CATCH_REQUIRE(runAttentionCase(1, 1, 1, 16, 128, causal));
    CATCH_REQUIRE(runAttentionCase(8, 8, 1, 1, 128, causal));
    // one query against a long history, the decode-step shape.
    CATCH_REQUIRE(runAttentionCase(8, 2, 1, 513, 128, causal));
  }
}

CATCH_TEST_CASE("test CUDA attention (grouped query ratios)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Every head count divisible by the key/value head count is a different grouping ratio, and
  // the ratio decides which key head each query head reads.
  for (bool causal : {false, true}) {
    CATCH_INFO("causal = " << causal);
    CATCH_REQUIRE(runAttentionCase(8, 1, 4, 12, 128, causal));  // all heads share one kv head
    CATCH_REQUIRE(runAttentionCase(8, 4, 4, 12, 128, causal));
    CATCH_REQUIRE(runAttentionCase(8, 8, 4, 12, 128, causal));  // no grouping
    CATCH_REQUIRE(runAttentionCase(12, 4, 4, 12, 128, causal));
  }
}

CATCH_TEST_CASE("test CUDA attention (causal boundaries)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Key lengths on and just off the kernel's tile boundaries, where the causal masking of the
  // final partial tile is easiest to get wrong.
  for (int keyLength : {127, 128, 129, 255, 256, 257}) {
    CATCH_INFO("keyLength = " << keyLength);
    CATCH_REQUIRE(runAttentionCase(8, 2, 1, keyLength, 128, true));
    CATCH_REQUIRE(runAttentionCase(8, 2, 4, keyLength, 128, true));
  }

  // a fresh prefill, where the query range is the whole key range.
  for (int length : {1, 2, 128, 129}) {
    CATCH_INFO("length = " << length);
    CATCH_REQUIRE(runAttentionCase(8, 2, length, length, 128, true));
  }
}

}  // namespace fl
