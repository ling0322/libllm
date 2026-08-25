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

#include "flint/cuda/flash_attn.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <algorithm>
#include <vector>

#include "flash.h"
#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {
namespace {

int roundMultiple(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

// Ported from the FlashAttention python bindings: pick the split count that fills the SMs best.
int chooseNumSplits(int numTiles, int numSm, int numKeyValueBlocks, int maxSplits) {
  if (numTiles >= 0.8f * numSm) return 1;

  maxSplits = std::min({maxSplits, numSm, numKeyValueBlocks});

  // A split only helps when it actually changes how many blocks each CTA walks.
  auto isEligible = [numKeyValueBlocks](int numSplits) {
    return numSplits == 1 ||
           ceilDiv(numKeyValueBlocks, numSplits) != ceilDiv(numKeyValueBlocks, numSplits - 1);
  };

  std::vector<float> efficiency;
  efficiency.reserve(maxSplits);
  float maxEfficiency = 0.0f;
  for (int numSplits = 1; numSplits <= maxSplits; ++numSplits) {
    if (!isEligible(numSplits)) {
      efficiency.push_back(0.0f);
      continue;
    }

    float numWaves = float(numTiles * numSplits) / numSm;
    float thisEfficiency = numWaves / std::ceil(numWaves);
    maxEfficiency = std::max(maxEfficiency, thisEfficiency);
    efficiency.push_back(thisEfficiency);
  }

  for (int numSplits = 1; numSplits <= maxSplits; ++numSplits) {
    if (!isEligible(numSplits)) continue;
    if (efficiency[numSplits - 1] >= 0.85f * maxEfficiency) return numSplits;
  }

  return 1;
}

bool isSupported(const Tensor &q, const Tensor &k, const Tensor &v) {
  if (q.getDType() != DType::kFloat16) return false;
  if (k.getDType() != DType::kFloat16 || v.getDType() != DType::kFloat16) return false;
  if (q.getDim() != 4 || k.getDim() != 4 || v.getDim() != 4) return false;

  int headDim = q.getShape(3);
  if (headDim != 64 && headDim != 128 && headDim != 256) return false;
  if (k.getShape(3) != headDim || v.getShape(3) != headDim) return false;

  // The kernels read the head dimension as packed vectors.
  if (q.getStride(3) != 1 || k.getStride(3) != 1 || v.getStride(3) != 1) return false;

  int numHeads = q.getShape(1);
  int numKeyValueHeads = k.getShape(1);
  if (numKeyValueHeads <= 0 || numHeads % numKeyValueHeads != 0) return false;
  if (v.getShape(1) != numKeyValueHeads || v.getShape(2) != k.getShape(2)) return false;

  return true;
}

bool isPagedSupported(const Tensor &q, const Tensor &keyCache, const Tensor &valueCache) {
  if (q.getDType() != DType::kFloat16) return false;
  if (keyCache.getDType() != DType::kFloat16 || valueCache.getDType() != DType::kFloat16)
    return false;
  if (q.getDim() != 3 || keyCache.getDim() != 4 || valueCache.getDim() != 4) return false;

  int headDim = q.getShape(2);
  if (headDim != 64 && headDim != 128 && headDim != 256) return false;
  if (keyCache.getShape(3) != headDim || valueCache.getShape(3) != headDim) return false;

  if (q.getStride(2) != 1 || keyCache.getStride(3) != 1 || valueCache.getStride(3) != 1)
    return false;

  int numHeads = q.getShape(1);
  int numKeyValueHeads = keyCache.getShape(2);
  if (numKeyValueHeads <= 0 || numHeads % numKeyValueHeads != 0) return false;
  if (valueCache.getShape(1) != keyCache.getShape(1)) return false;
  if (valueCache.getShape(2) != numKeyValueHeads) return false;

  return true;
}

}  // namespace

Tensor flashAttention(Tensor q, Tensor k, Tensor v, bool causal) {
  if (!isSupported(q, k, v)) return Tensor();

  int batchSize = q.getShape(0);
  int numHeads = q.getShape(1);
  int queryLength = q.getShape(2);
  int headDim = q.getShape(3);
  int numKeyValueHeads = k.getShape(1);
  int keyValueLength = k.getShape(2);

  Tensor output = createCudaTensorHalf({batchSize, numHeads, queryLength, headDim});
  Tensor softmaxLse = createCudaTensorFloat({batchSize, numHeads, queryLength});

  FLASH_NAMESPACE::Flash_fwd_params params{};

  params.q_ptr = getDataPtrCuda<half>(q);
  params.k_ptr = getDataPtrCuda<half>(k);
  params.v_ptr = getDataPtrCuda<half>(v);
  params.o_ptr = getDataPtrCuda<half>(output);
  params.softmax_lse_ptr = getDataPtrCuda<float>(softmaxLse);

  params.q_batch_stride = q.getStride(0);
  params.q_head_stride = q.getStride(1);
  params.q_row_stride = q.getStride(2);
  params.k_batch_stride = k.getStride(0);
  params.k_head_stride = k.getStride(1);
  params.k_row_stride = k.getStride(2);
  params.v_batch_stride = v.getStride(0);
  params.v_head_stride = v.getStride(1);
  params.v_row_stride = v.getStride(2);
  params.o_batch_stride = output.getStride(0);
  params.o_head_stride = output.getStride(1);
  params.o_row_stride = output.getStride(2);

  params.b = batchSize;
  params.h = numHeads;
  params.h_k = numKeyValueHeads;
  params.h_h_k_ratio = numHeads / numKeyValueHeads;
  params.d = headDim;
  params.d_rounded = roundMultiple(headDim, 32);
  params.seqlen_q = queryLength;
  params.seqlen_k = keyValueLength;
  params.seqlen_q_rounded = roundMultiple(queryLength, 128);
  params.seqlen_k_rounded = roundMultiple(keyValueLength, 128);
  params.total_q = batchSize * queryLength;

  float scale = 1.0f / sqrtf(1.0f * headDim);
  params.scale_softmax = scale;
  params.scale_softmax_log2 = scale * static_cast<float>(M_LOG2E);

  // Dropout is compiled out of the kernels, so every activation is kept.
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = scale;

  // The causal mask is aligned to the bottom right of the score matrix.
  params.is_causal = causal;
  params.window_size_left = causal ? keyValueLength : -1;
  params.window_size_right = causal ? 0 : -1;

  params.is_seqlens_k_cumulative = true;

  // The split kernel uses a fixed 64 row block and walks the keys in these blocks.
  int keyValueBlock = headDim <= 64 ? 256 : (headDim <= 128 ? 128 : 64);
  int numTiles = batchSize * numHeads * ceilDiv(queryLength, 64);
  int numSm = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  int numSplits =
      chooseNumSplits(numTiles, numSm * 2, ceilDiv(keyValueLength, keyValueBlock), 128);

  Tensor outputAccum;
  Tensor softmaxLseAccum;
  if (numSplits > 1) {
    outputAccum = createCudaTensorFloat(
        {numSplits, batchSize, numHeads, queryLength, params.d_rounded});
    softmaxLseAccum = createCudaTensorFloat({numSplits, batchSize, numHeads, queryLength});
    params.oaccum_ptr = getDataPtrCuda<float>(outputAccum);
    params.softmax_lseaccum_ptr = getDataPtrCuda<float>(softmaxLseAccum);
  }
  params.num_splits = numSplits;

  if (!FLASH_NAMESPACE::run_mha_fwd(params, nullptr)) return Tensor();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return output;
}

Tensor pagedFlashAttention(
    Tensor q,
    Tensor keyCache,
    Tensor valueCache,
    Tensor blockTable,
    Tensor cuSeqlensQ,
    Tensor seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal) {
  if (!isPagedSupported(q, keyCache, valueCache)) return Tensor();
  CHECK(blockTable.getDType() == DType::kInt32);
  CHECK(cuSeqlensQ.getDType() == DType::kInt32);
  CHECK(seqlensK.getDType() == DType::kInt32);
  CHECK(blockTable.getDim() == 2 && cuSeqlensQ.getDim() == 1 && seqlensK.getDim() == 1);
  CHECK(maxQLen > 0 && maxKLen > 0);

  int totalQ = q.getShape(0);
  int numHeads = q.getShape(1);
  int headDim = q.getShape(2);
  int blockSize = keyCache.getShape(1);
  int numKeyValueHeads = keyCache.getShape(2);
  int numSequences = seqlensK.getShape(0);
  CHECK(cuSeqlensQ.getShape(0) == numSequences + 1);
  CHECK(blockTable.getShape(0) == numSequences);
  CHECK(maxKLen <= blockTable.getShape(1) * blockSize);

  Tensor output = createCudaTensorHalf({totalQ, numHeads, headDim});
  Tensor softmaxLse = createCudaTensorFloat({numHeads, totalQ});

  FLASH_NAMESPACE::Flash_fwd_params params{};

  params.q_ptr = getDataPtrCuda<half>(q);
  params.k_ptr = getDataPtrCuda<half>(keyCache);
  params.v_ptr = getDataPtrCuda<half>(valueCache);
  params.o_ptr = getDataPtrCuda<half>(output);
  params.softmax_lse_ptr = getDataPtrCuda<float>(softmaxLse);

  params.q_row_stride = q.getStride(0);
  params.q_head_stride = q.getStride(1);
  // The kernel reaches a block through block_table[i] * k_batch_stride, so a block plays the role
  // its batch element plays in the dense kernel.
  params.k_batch_stride = keyCache.getStride(0);
  params.k_row_stride = keyCache.getStride(1);
  params.k_head_stride = keyCache.getStride(2);
  params.v_batch_stride = valueCache.getStride(0);
  params.v_row_stride = valueCache.getStride(1);
  params.v_head_stride = valueCache.getStride(2);
  params.o_row_stride = output.getStride(0);
  params.o_head_stride = output.getStride(1);

  params.cu_seqlens_q = getDataPtrCuda<IntType>(cuSeqlensQ);
  // seqused_k hands the kernel each sequence length directly, so cu_seqlens_k stays null and the
  // caller never has to build a cumulative array for keys it cannot address contiguously anyway.
  params.seqused_k = getDataPtrCuda<IntType>(seqlensK);
  params.is_seqlens_k_cumulative = true;

  params.block_table = getDataPtrCuda<IntType>(blockTable);
  params.block_table_batch_stride = blockTable.getStride(0);
  params.page_block_size = blockSize;

  params.b = numSequences;
  params.h = numHeads;
  params.h_k = numKeyValueHeads;
  params.h_h_k_ratio = numHeads / numKeyValueHeads;
  params.d = headDim;
  params.d_rounded = roundMultiple(headDim, 32);
  // These size the grid and the key loop for the whole batch, so they must not be below any
  // sequence's own length; cu_seqlens_q and seqused_k carry the per-sequence lengths. Too large
  // only wastes blocks that exit early, too small silently drops queries or keys.
  params.seqlen_q = maxQLen;
  params.seqlen_k = maxKLen;
  params.seqlen_q_rounded = roundMultiple(maxQLen, 128);
  params.seqlen_k_rounded = roundMultiple(maxKLen, 128);
  params.total_q = totalQ;
  // The LSE of a packed batch is <float>(nHead, totalQLen), not padded per sequence.
  params.unpadded_lse = true;

  float scale = 1.0f / sqrtf(1.0f * headDim);
  params.scale_softmax = scale;
  params.scale_softmax_log2 = scale * static_cast<float>(M_LOG2E);

  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = scale;

  // The causal mask is aligned to the bottom right of the score matrix.
  params.is_causal = causal;
  params.window_size_left = causal ? maxKLen : -1;
  params.window_size_right = causal ? 0 : -1;

  // Splitting would combine the partial outputs through o_batch_stride, which a packed batch has
  // no meaningful value for, so the split kernel runs with a single split.
  params.num_splits = 1;

  if (!FLASH_NAMESPACE::run_mha_fwd(params, nullptr, /*force_split_kernel=*/true)) return Tensor();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return output;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
