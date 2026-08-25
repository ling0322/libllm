// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include <algorithm>
#include <vector>

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

/// One sequence of a packed batch, and the blocks it owns.
struct Sequence {
  std::vector<int> blockIds;
  int queryLength;
  int keyLength;
};

/// Read back the first keyLength tokens a sequence owns, so the reference attends to exactly the
/// values the kernel has to walk the block table to reach.
Tensor gatherFromPool(const Tensor &pool, const Sequence &sequence) {
  int blockSize = pool.getShape(1);
  Tensor gathered;
  for (int offset = 0; offset < sequence.keyLength; offset += blockSize) {
    int length = std::min(blockSize, sequence.keyLength - offset);
    Tensor part = pool.subtensor(sequence.blockIds[offset / blockSize]).slice(0, {0, length});
    gathered = offset == 0 ? part : F::cat(gathered, part, 0);
  }

  return gathered;
}

/// Attention of one sequence against its own keys, computed by the dense operator.
Tensor referenceAttention(
    const Tensor &query,
    const Tensor &key,
    const Tensor &value,
    bool causal) {
  // The dense operator wants <float>(batch, head, length, headDim).
  Tensor x = F::attention(
      query.unsqueeze(0).transpose(1, 2),
      key.unsqueeze(0).transpose(1, 2),
      value.unsqueeze(0).transpose(1, 2),
      causal);

  return F::contiguous(x.subtensor(0).transpose(0, 1));
}

bool runCase(
    int numHeads,
    int numKeyValueHeads,
    int headDim,
    int blockSize,
    int numBlocks,
    const std::vector<Sequence> &sequences,
    bool causal) {
  int numSequences = static_cast<int>(sequences.size());
  int maxNumBlocks = 0;
  int totalQueryLength = 0;
  int maxQueryLength = 0;
  int maxKeyLength = 0;
  for (const Sequence &sequence : sequences) {
    maxNumBlocks = std::max(maxNumBlocks, static_cast<int>(sequence.blockIds.size()));
    totalQueryLength += sequence.queryLength;
    maxQueryLength = std::max(maxQueryLength, sequence.queryLength);
    maxKeyLength = std::max(maxKeyLength, sequence.keyLength);
  }

  Tensor keyPool = F::rand({numBlocks, blockSize, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor valuePool = F::rand({numBlocks, blockSize, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor query = F::rand({totalQueryLength, numHeads, headDim}, DType::kFloat);

  std::vector<IntType> blockTableData(numSequences * maxNumBlocks, 0);
  std::vector<IntType> cuSeqlensQData(numSequences + 1, 0);
  std::vector<IntType> seqlensKData(numSequences, 0);
  for (int i = 0; i < numSequences; ++i) {
    const Sequence &sequence = sequences[i];
    std::copy(
        sequence.blockIds.begin(),
        sequence.blockIds.end(),
        blockTableData.begin() + i * maxNumBlocks);
    cuSeqlensQData[i + 1] = cuSeqlensQData[i] + sequence.queryLength;
    seqlensKData[i] = sequence.keyLength;
  }

  Tensor blockTable = Tensor::create<IntType>({numSequences, maxNumBlocks}, blockTableData);
  Tensor cuSeqlensQ = Tensor::create<IntType>({numSequences + 1}, cuSeqlensQData);
  Tensor seqlensK = Tensor::create<IntType>({numSequences}, seqlensKData);

  Tensor x = F::pagedAttention(
      toCuda(query),
      toCuda(keyPool),
      toCuda(valuePool),
      F::to(Device::getCuda(), blockTable),
      F::to(Device::getCuda(), cuSeqlensQ),
      F::to(Device::getCuda(), seqlensK),
      maxQueryLength,
      maxKeyLength,
      causal);

  if (x.getShape() != std::vector<int>{totalQueryLength, numHeads, headDim}) return false;
  Tensor output = toCpu(x);

  for (int i = 0; i < numSequences; ++i) {
    const Sequence &sequence = sequences[i];
    int begin = cuSeqlensQData[i];
    int end = begin + sequence.queryLength;

    Tensor expected = referenceAttention(
        query.slice(0, {begin, end}),
        gatherFromPool(keyPool, sequence),
        gatherFromPool(valuePool, sequence),
        causal);

    if (!F::allClose(output.slice(0, {begin, end}), expected, 5e-3f)) return false;
  }

  return true;
}

}  // namespace

CATCH_TEST_CASE("test CUDA pagedAttention", "[op][cuda][flash_attn]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The blocks are scrambled and non-contiguous, so a kernel that walks the pool linearly instead
  // of through the block table reads another sequence's keys and cannot match the reference.
  std::vector<Sequence> mixedBatch = {
      {{5, 1}, 4, 300},
      {{3}, 1, 200},
      {{7, 0, 2}, 300, 600},
  };

  for (bool causal : {false, true}) {
    CATCH_REQUIRE(runCase(8, 2, 128, 256, 8, mixedBatch, causal));

    // one sequence, so the batch dimension of the block table degenerates.
    CATCH_REQUIRE(runCase(8, 2, 128, 256, 8, {{{6, 3}, 8, 400}}, causal));

    // multi-head attention takes the h_h_k_ratio == 1 path.
    CATCH_REQUIRE(runCase(4, 4, 128, 256, 8, mixedBatch, causal));
  }
}

CATCH_TEST_CASE("test CUDA pagedAttention (block boundaries)", "[op][cuda][flash_attn]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Key lengths that land exactly on a block and on a key tile, then one token either side, which
  // is where the tail masking and the block walk are easiest to get wrong.
  std::vector<Sequence> aligned = {
      {{4, 2}, 1, 256},
      {{1, 6}, 1, 512},
      {{7, 0}, 1, 128},
  };
  std::vector<Sequence> offByOne = {
      {{4, 2}, 1, 255},
      {{1, 6}, 1, 257},
      {{7, 0}, 1, 511},
  };

  for (bool causal : {false, true}) {
    CATCH_REQUIRE(runCase(8, 2, 128, 256, 8, aligned, causal));
    CATCH_REQUIRE(runCase(8, 2, 128, 256, 8, offByOne, causal));
  }

  // A fresh prefill has no history, so the whole key range is also the query range.
  CATCH_REQUIRE(runCase(8, 2, 128, 256, 8, {{{2}, 256, 256}}, true));
  CATCH_REQUIRE(runCase(8, 2, 128, 256, 8, {{{2, 5}, 300, 300}}, true));
}

CATCH_TEST_CASE("test CUDA pagedAttention (head dims)", "[op][cuda][flash_attn]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The key tile is 256, 128 and 64 wide for head dims 64, 128 and 256, so a 256-token block holds
  // one, two and four tiles. Each ratio exercises different block-walk arithmetic.
  std::vector<Sequence> batch = {
      {{5, 1}, 4, 300},
      {{3}, 1, 200},
      {{7, 0, 2}, 64, 600},
  };

  for (bool causal : {false, true}) {
    CATCH_REQUIRE(runCase(8, 2, 64, 256, 8, batch, causal));
    CATCH_REQUIRE(runCase(8, 2, 256, 256, 8, batch, causal));
  }
}

}  // namespace fl
