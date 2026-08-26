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
#include <mma.h>

#include <limits>

#include "flint/cuda/common.h"
#include "flint/cuda/gated_delta_net_mma.h"
#include "flint/cuda/gated_delta_net_wmma.h"
#include "flint/cuda/triangular_solve.h"

namespace fl {
namespace op {
namespace cuda {

using namespace nvcuda;

/// The three-launch implementation: build every chunk's system, solve the whole batch of them
/// through triangularSolveInplace, then scan each sequence's chunks in order. Nothing here knows about a
/// sequence as a whole except the scan, so the first two launches get one block per (chunk, head)
/// and fill the machine however short the batch is.
namespace chunked {
namespace {

/// The chunk length, which is also the size of the triangular system each chunk solves.
constexpr int kChunk = 64;

/// Value columns one scan block owns. The block keeps that slice of the (D, D) state in shared
/// memory for the whole sequence, which is what bounds it: D * kValueBlock floats.
constexpr int kValueBlock = 32;

/// The reduction tile the GEMMs stage through shared memory.
constexpr int kDimTile = 32;

/// The row stride of a staged tile, in elements. A warp reads one column of a tile across rows
/// four apart, and an even stride puts every one of those rows in the same shared memory bank; the
/// odd padding is what spreads them out. It is worth several times the kernel's runtime.
constexpr int kTileStride = kDimTile + 1;

constexpr int kTableThreads = 256;
constexpr int kBuildThreads = 256;
constexpr int kScanThreads = 128;

/// Each thread owns a 4 by 4 tile of whatever output it is accumulating. The build kernel spreads
/// its 256 threads over that as 16 row groups by 16 column groups, covering a kChunk by kChunk
/// matrix; the scan kernel spreads its 128 as 16 by 8, covering kChunk by kValueBlock.
constexpr int kTileM = 4;
constexpr int kTileN = 4;

/// The tensor core tile. A warp computes one 16 by 16 block of a score matrix at a time, over 16
/// elements of the head dimension per instruction.
constexpr int kMmaM = 16;
constexpr int kMmaN = 16;
constexpr int kMmaK = 16;

/// The slice of the head dimension staged for those GEMMs, and its row stride. The padding keeps
/// the 128-bit loads a fragment issues off one bank while staying a multiple of eight, which is
/// what load_matrix_sync needs of a leading dimension.
constexpr int kMmaDimTile = 32;
constexpr int kMmaLd = kMmaDimTile + 8;

/// The epilogue stages an accumulator per warp rather than the whole score matrix per block: it is
/// a third of the shared memory, which is a third more CTAs resident, and a warp reading back what
/// it wrote itself needs no block-wide barrier. The padding is four rather than one because
/// store_matrix_sync wants a leading dimension that is a multiple of four floats.
constexpr int kTileLd = kMmaN + 4;

constexpr int kNumWarps = kBuildThreads / 32;
constexpr int kTilesPerSide = kChunk / kMmaM;
constexpr int kTilesPerWarp = kTilesPerSide * kTilesPerSide / kNumWarps;

}  // namespace

/// Turn the sequence lengths into a flat list of chunks. Slots past the last real chunk keep a
/// length of zero: the batched solve runs over all of them, so the build kernel gives those an
/// identity system rather than leaving the batch ragged.
__global__ void chunkTableKernel(
    const int32_t *__restrict__ cuSeqlens,
    int32_t *__restrict__ chunkStart,
    int32_t *__restrict__ chunkLen,
    int32_t *__restrict__ seqChunkBegin,
    int32_t *__restrict__ seqChunkCount,
    int numSeq,
    int numSlots) {
  for (int slot = threadIdx.x; slot < numSlots; slot += kTableThreads) {
    chunkStart[slot] = 0;
    chunkLen[slot] = 0;
  }

  if (threadIdx.x == 0) {
    int begin = 0;
    for (int s = 0; s < numSeq; ++s) {
      int len = cuSeqlens[s + 1] - cuSeqlens[s];
      int count = (len + kChunk - 1) / kChunk;
      seqChunkBegin[s] = begin;
      seqChunkCount[s] = count;
      begin += count;
    }
  }
  __syncthreads();

  for (int s = threadIdx.x; s < numSeq; s += kTableThreads) {
    int base = seqChunkBegin[s];
    int tokenBegin = cuSeqlens[s];
    int len = cuSeqlens[s + 1] - tokenBegin;
    for (int c = 0; c < seqChunkCount[s]; ++c) {
      chunkStart[base + c] = tokenBegin + c * kChunk;
      chunkLen[base + c] = min(kChunk, len - c * kChunk);
    }
  }
}

/// Everything about a chunk that does not depend on the incoming state: the triangular system
/// (I + A), the two right hand sides packed side by side, the decayed query-key scores, and the two
/// decay windows the scan needs. One block per (chunk, value head).
__global__ void buildChunkKernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    const float *__restrict__ g,
    const float *__restrict__ beta,
    const int32_t *__restrict__ chunkStart,
    const int32_t *__restrict__ chunkLen,
    float *__restrict__ a,
    half *__restrict__ rhs,
    half *__restrict__ mmat,
    float *__restrict__ cexp,
    float *__restrict__ clast,
    int numKHead,
    int numVHead,
    int headDim,
    int headRatio) {
  __shared__ float sCum[kChunk];
  __shared__ float sCexp[kChunk];
  __shared__ float sBeta[kChunk];
  __shared__ half sK[kChunk * kMmaLd];
  __shared__ half sQ[kChunk * kMmaLd];
  __shared__ float sTileA[kNumWarps * kMmaM * kTileLd];
  __shared__ float sTileM[kNumWarps * kMmaM * kTileLd];

  const int slot = blockIdx.x;
  const int head = blockIdx.y;
  const int kHead = head / headRatio;
  const int tid = threadIdx.x;
  const int len = chunkLen[slot];

  const int64_t matOffset = (static_cast<int64_t>(slot) * numVHead + head) * kChunk * kChunk;
  float *aOut = a + matOffset;
  half *mOut = mmat + matOffset;

  if (len == 0) {
    for (int idx = tid; idx < kChunk * kChunk; idx += kBuildThreads) {
      aOut[idx] = (idx / kChunk == idx % kChunk) ? 1.0f : 0.0f;
    }
    return;
  }

  const int start = chunkStart[slot];
  const int64_t vecOffset = (static_cast<int64_t>(slot) * numVHead + head) * kChunk;

  // The cumulative log decay, kept in the log domain. Every decay below is exp of a difference of
  // two of these, so a long run of strongly decaying steps never has to divide two exponentials
  // that have both underflowed.
  if (tid == 0) {
    float cum = 0.0f;
    for (int i = 0; i < len; ++i) {
      cum += g[static_cast<int64_t>(start + i) * numVHead + head];
      sCum[i] = cum;
    }
  }
  __syncthreads();

  const float cumLast = sCum[len - 1];
  for (int i = tid; i < len; i += kBuildThreads) {
    sCexp[i] = expf(sCum[i]);
    sBeta[i] = beta[static_cast<int64_t>(start + i) * numVHead + head];
    cexp[vecOffset + i] = sCexp[i];
    clast[vecOffset + i] = expf(cumLast - sCum[i]);
  }
  __syncthreads();

  // A_ij = k_i . k_j and M_ij = q_i . k_j, both before their decays. These are the two products
  // the tensor cores do: half operands into float accumulators, which is exactly what an HMMA is,
  // and q and k arrive in half anyway so nothing is given up by it. Each warp owns a 16 by 16 block
  // of both score matrices and walks the head dimension a staged slice at a time.
  const int warp = tid / 32;
  wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> accA[kTilesPerWarp];
  wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> accM[kTilesPerWarp];
#pragma unroll
  for (int t = 0; t < kTilesPerWarp; ++t) {
    wmma::fill_fragment(accA[t], 0.0f);
    wmma::fill_fragment(accM[t], 0.0f);
  }

  for (int d0 = 0; d0 < headDim; d0 += kMmaDimTile) {
    __syncthreads();
    for (int idx = tid; idx < kChunk * kMmaDimTile; idx += kBuildThreads) {
      int r = idx / kMmaDimTile;
      int c = idx - r * kMmaDimTile;
      bool ok = r < len && d0 + c < headDim;
      int64_t off = (static_cast<int64_t>(start + r) * numKHead + kHead) * headDim + d0 + c;
      sK[r * kMmaLd + c] = ok ? k[off] : __float2half(0.0f);
      sQ[r * kMmaLd + c] = ok ? q[off] : __float2half(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t) {
      int tile = warp * kTilesPerWarp + t;
      int ti = (tile / kTilesPerSide) * kMmaM;
      int tj = (tile % kTilesPerSide) * kMmaN;

      for (int kk = 0; kk < kMmaDimTile; kk += kMmaK) {
        // B is K transposed, so it is the same buffer read down its columns: element (d, j) of the
        // transpose is element (j, d) of K, which is what a column major fragment over the same
        // leading dimension reads.
        wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, half, wmma::row_major> aK;
        wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, half, wmma::row_major> aQ;
        wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, half, wmma::col_major> bK;
        wmma::load_matrix_sync(aK, sK + ti * kMmaLd + kk, kMmaLd);
        wmma::load_matrix_sync(aQ, sQ + ti * kMmaLd + kk, kMmaLd);
        wmma::load_matrix_sync(bK, sK + tj * kMmaLd + kk, kMmaLd);
        wmma::mma_sync(accA[t], aK, bK, accA[t]);
        wmma::mma_sync(accM[t], aQ, bK, accM[t]);
      }
    }
  }

  // An accumulator fragment's element order is not something the program is allowed to know, so the
  // decays cannot be folded into it: the scores go through shared memory and are scaled there. Both
  // matrices are staged at once so that the exponential each element needs is taken once, not
  // twice, and a warp only ever reads back the tile it wrote.
  const int lane = tid & 31;
  float *tileA = sTileA + warp * kMmaM * kTileLd;
  float *tileM = sTileM + warp * kMmaM * kTileLd;

#pragma unroll
  for (int t = 0; t < kTilesPerWarp; ++t) {
    int tile = warp * kTilesPerWarp + t;
    int ti = (tile / kTilesPerSide) * kMmaM;
    int tj = (tile % kTilesPerSide) * kMmaN;

    wmma::store_matrix_sync(tileA, accA[t], kTileLd, wmma::mem_row_major);
    wmma::store_matrix_sync(tileM, accM[t], kTileLd, wmma::mem_row_major);
    __syncwarp();

    for (int e = lane; e < kMmaM * kMmaN; e += 32) {
      int r = e / kMmaN;
      int c = e - r * kMmaN;
      int i = ti + r;
      int j = tj + c;

      float av = 0.0f;
      float mv = 0.0f;
      if (i < len && j <= i) {
        float decay = expf(sCum[i] - sCum[j]);
        if (j < i) av = sBeta[i] * decay * tileA[r * kTileLd + c];
        mv = decay * tileM[r * kTileLd + c];
      }

      // The diagonal is the identity part of (I + A), for the padding rows as much as the real
      // ones, so the batched solve leaves a short chunk's unused rows alone.
      if (i == j) av = 1.0f;
      aOut[i * kChunk + j] = av;
      mOut[i * kChunk + j] = __float2half(mv);
    }
    __syncwarp();
  }

  // The two right hand sides: beta * V, whose solution is the chunk's writes against a zero state,
  // and diag(beta exp(c)) K, whose solution is what the incoming state is multiplied by. Row by row
  // rather than over the flattened tile: the row width is a runtime value, so a flat loop would pay
  // an integer division per element, and there are 2 * D of them per row.
  const int width = 2 * headDim;
  half *rOut = rhs + (static_cast<int64_t>(slot) * numVHead + head) * kChunk * width;
  for (int i = 0; i < kChunk; ++i) {
    half *row = rOut + static_cast<int64_t>(i) * width;
    if (i >= len) {
      for (int c = tid; c < width; c += kBuildThreads) row[c] = __float2half(0.0f);
      continue;
    }

    const half *vRow = v + (static_cast<int64_t>(start + i) * numVHead + head) * headDim;
    const half *kRow = k + (static_cast<int64_t>(start + i) * numKHead + kHead) * headDim;
    float bv = sBeta[i];
    float bk = sBeta[i] * sCexp[i];
    for (int c = tid; c < headDim; c += kBuildThreads) {
      row[c] = __float2half(bv * __half2float(vRow[c]));
      row[headDim + c] = __float2half(bk * __half2float(kRow[c]));
    }
  }
}

/// Carry one (sequence, value head, slice of the value dimension) through its chunks in order. The
/// state slice lives in shared memory for the whole sequence; everything else streams past it a
/// chunk at a time. Splitting the value dimension is what keeps that slice small enough to hold:
/// the whole (D, D) state would not fit, a (D, kValueBlock) column of it does.
__global__ void scanKernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ rhs,
    const half *__restrict__ mmat,
    const float *__restrict__ cexp,
    const float *__restrict__ clast,
    const int32_t *__restrict__ chunkStart,
    const int32_t *__restrict__ chunkLen,
    const int32_t *__restrict__ seqChunkBegin,
    const int32_t *__restrict__ seqChunkCount,
    const int32_t *__restrict__ stateSlots,
    float *__restrict__ state,
    half *__restrict__ out,
    int numKHead,
    int numVHead,
    int headDim,
    int headRatio) {
  // Only the state stays in float. u, W and the query, key and score tiles are all GEMM operands
  // that are consumed into float accumulators, so holding them in half halves what this kernel
  // reads and, for u, what it keeps resident for the whole chunk.
  extern __shared__ char smemRaw[];
  float *sState = reinterpret_cast<float *>(smemRaw);
  half *sUhat = reinterpret_cast<half *>(sState + headDim * kValueBlock);
  half *sW = sUhat + kChunk * kValueBlock;
  half *sH = sW + kChunk * kTileStride;

  __shared__ float sCexp[kChunk];
  __shared__ float sClast[kChunk];

  const int numSlices = headDim / kValueBlock;
  const int seq = blockIdx.x;
  const int head = blockIdx.y / numSlices;
  const int col0 = (blockIdx.y - head * numSlices) * kValueBlock;
  const int kHead = head / headRatio;
  const int tid = threadIdx.x;
  const int ty = tid / 8;
  const int tx = tid % 8;
  const int col = col0 + tx * kTileN;

  // The state is reached through the slot mapping rather than by position in the batch, so the
  // pool this block reads and writes belongs to the sequence rather than to the launch.
  const int64_t stateOffset =
      (static_cast<int64_t>(stateSlots[seq]) * numVHead + head) * headDim * headDim;
  for (int idx = tid; idx < headDim * kValueBlock; idx += kScanThreads) {
    int m = idx / kValueBlock;
    int c = idx - m * kValueBlock;
    sState[idx] = state[stateOffset + static_cast<int64_t>(m) * headDim + col0 + c];
  }

  const int width = 2 * headDim;
  const int chunkBegin = seqChunkBegin[seq];
  const int chunkCount = seqChunkCount[seq];

  for (int ci = 0; ci < chunkCount; ++ci) {
    const int slot = chunkBegin + ci;
    const int len = chunkLen[slot];
    const int start = chunkStart[slot];
    const int64_t vecOffset = (static_cast<int64_t>(slot) * numVHead + head) * kChunk;
    const half *rp = rhs + (static_cast<int64_t>(slot) * numVHead + head) * kChunk * width;
    const half *mp = mmat + (static_cast<int64_t>(slot) * numVHead + head) * kChunk * kChunk;

    __syncthreads();
    for (int i = tid; i < len; i += kScanThreads) {
      sCexp[i] = cexp[vecOffset + i];
      sClast[i] = clast[vecOffset + i];
    }
    __syncthreads();

    // u = U - W S, the chunk's writes once the incoming state is accounted for.
    float acc[kTileM][kTileN];
#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
      int i = ty * kTileM + r;
#pragma unroll
      for (int b = 0; b < kTileN; ++b) {
        acc[r][b] = (i < len) ? __half2float(rp[i * width + col + b]) : 0.0f;
      }
    }

    for (int m0 = 0; m0 < headDim; m0 += kDimTile) {
      __syncthreads();
      for (int idx = tid; idx < kChunk * kDimTile; idx += kScanThreads) {
        int i = idx / kDimTile;
        int c = idx - i * kDimTile;
        sW[i * kTileStride + c] = (i < len) ? rp[i * width + headDim + m0 + c]
                                            : __float2half(0.0f);
      }
      __syncthreads();

      for (int c = 0; c < kDimTile; ++c) {
        float sv[kTileN];
#pragma unroll
        for (int b = 0; b < kTileN; ++b) sv[b] = sState[(m0 + c) * kValueBlock + tx * kTileN + b];

#pragma unroll
        for (int r = 0; r < kTileM; ++r) {
          float wv = __half2float(sW[(ty * kTileM + r) * kTileStride + c]);
#pragma unroll
          for (int b = 0; b < kTileN; ++b) acc[r][b] -= wv * sv[b];
        }
      }
    }

    __syncthreads();
#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
#pragma unroll
      for (int b = 0; b < kTileN; ++b) {
        sUhat[(ty * kTileM + r) * kValueBlock + tx * kTileN + b] = __float2half(acc[r][b]);
      }
    }

    // o_i = exp(c_i) q_i S + sum_{j<=i} M_ij u_j, both against the state this chunk came in with.
    float accO[kTileM][kTileN] = {};
    for (int m0 = 0; m0 < headDim; m0 += kDimTile) {
      __syncthreads();
      for (int idx = tid; idx < kChunk * kDimTile; idx += kScanThreads) {
        int i = idx / kDimTile;
        int c = idx - i * kDimTile;
        int64_t off = (static_cast<int64_t>(start + i) * numKHead + kHead) * headDim + m0 + c;
        sH[i * kTileStride + c] = (i < len) ? q[off] : __float2half(0.0f);
      }
      __syncthreads();

      for (int c = 0; c < kDimTile; ++c) {
        float sv[kTileN];
#pragma unroll
        for (int b = 0; b < kTileN; ++b) sv[b] = sState[(m0 + c) * kValueBlock + tx * kTileN + b];

#pragma unroll
        for (int r = 0; r < kTileM; ++r) {
          float qv = __half2float(sH[(ty * kTileM + r) * kTileStride + c]);
#pragma unroll
          for (int b = 0; b < kTileN; ++b) accO[r][b] += qv * sv[b];
        }
      }
    }

#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
      int i = ty * kTileM + r;
      float scale = (i < len) ? sCexp[i] : 0.0f;
#pragma unroll
      for (int b = 0; b < kTileN; ++b) accO[r][b] *= scale;
    }

    for (int j0 = 0; j0 < len; j0 += kDimTile) {
      __syncthreads();
      for (int idx = tid; idx < kChunk * kDimTile; idx += kScanThreads) {
        int i = idx / kDimTile;
        int c = idx - i * kDimTile;
        sH[i * kTileStride + c] = (i < len && j0 + c < len) ? mp[i * kChunk + j0 + c]
                                                             : __float2half(0.0f);
      }
      __syncthreads();

      int cols = min(kDimTile, len - j0);
      for (int c = 0; c < cols; ++c) {
        float uv[kTileN];
#pragma unroll
        for (int b = 0; b < kTileN; ++b) {
          uv[b] = __half2float(sUhat[(j0 + c) * kValueBlock + tx * kTileN + b]);
        }

#pragma unroll
        for (int r = 0; r < kTileM; ++r) {
          float mv = __half2float(sH[(ty * kTileM + r) * kTileStride + c]);
#pragma unroll
          for (int b = 0; b < kTileN; ++b) accO[r][b] += mv * uv[b];
        }
      }
    }

#pragma unroll
    for (int r = 0; r < kTileM; ++r) {
      int i = ty * kTileM + r;
      if (i >= len) continue;

      int64_t off = (static_cast<int64_t>(start + i) * numVHead + head) * headDim + col;
#pragma unroll
      for (int b = 0; b < kTileN; ++b) out[off + b] = __float2half(accO[r][b]);
    }

    // S <- exp(c_last) S + sum_j exp(c_last - c_j) k_j u_j^T. The output has D rows against the
    // chunk's kChunk, so the row groups take D a band at a time.
    const float gamma = sCexp[len - 1];
    for (int m0 = 0; m0 < headDim; m0 += kChunk) {
      __syncthreads();
      float accS[kTileM][kTileN];
#pragma unroll
      for (int r = 0; r < kTileM; ++r) {
        int m = m0 + ty * kTileM + r;
#pragma unroll
        for (int b = 0; b < kTileN; ++b) {
          accS[r][b] = (m < headDim) ? gamma * sState[m * kValueBlock + tx * kTileN + b] : 0.0f;
        }
      }

      for (int j0 = 0; j0 < len; j0 += kDimTile) {
        __syncthreads();
        // sH[jj][mm] is k of token j0 + jj at row m0 + mm, the transpose of how the other tiles
        // stage, because here the head dimension indexes the output rather than the reduction. The
        // rows a warp reads out of it are adjacent rather than four apart, so this layout needs no
        // padding and reuses the same buffer.
        for (int idx = tid; idx < kDimTile * kChunk; idx += kScanThreads) {
          int jj = idx / kChunk;
          int mm = idx - jj * kChunk;
          bool ok = j0 + jj < len && m0 + mm < headDim;
          int64_t off =
              (static_cast<int64_t>(start + j0 + jj) * numKHead + kHead) * headDim + m0 + mm;
          sH[idx] = ok ? k[off] : __float2half(0.0f);
        }
        __syncthreads();

        int cols = min(kDimTile, len - j0);
        for (int c = 0; c < cols; ++c) {
          float uv[kTileN];
#pragma unroll
          for (int b = 0; b < kTileN; ++b) {
            uv[b] = __half2float(sUhat[(j0 + c) * kValueBlock + tx * kTileN + b]);
          }

          float decay = sClast[j0 + c];
#pragma unroll
          for (int r = 0; r < kTileM; ++r) {
            float kv = decay * __half2float(sH[c * kChunk + ty * kTileM + r]);
#pragma unroll
            for (int b = 0; b < kTileN; ++b) accS[r][b] += kv * uv[b];
          }
        }
      }

      __syncthreads();
#pragma unroll
      for (int r = 0; r < kTileM; ++r) {
        int m = m0 + ty * kTileM + r;
        if (m >= headDim) continue;
#pragma unroll
        for (int b = 0; b < kTileN; ++b) {
          sState[m * kValueBlock + tx * kTileN + b] = accS[r][b];
        }
      }
    }
  }

  __syncthreads();
  for (int idx = tid; idx < headDim * kValueBlock; idx += kScanThreads) {
    int m = idx / kValueBlock;
    int c = idx - m * kValueBlock;
    state[stateOffset + static_cast<int64_t>(m) * headDim + col0 + c] = sState[idx];
  }
}

void run(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &g,
    const Tensor &beta,
    const Tensor &cuSeqlens,
    const Tensor &stateSlots,
    Tensor &state,
    Tensor &o,
    int numTokens,
    int numKHead,
    int numVHead,
    int headDim,
    int numSeq) {
  CHECK(headDim % kValueBlock == 0 && headDim % kDimTile == 0)
      << "the chunked gatedDeltaNetPrefill needs a head dimension that is a multiple of "
      << kValueBlock;

  // Every sequence wastes at most the tail of one chunk, so this bounds the real chunk count. The
  // slots past it carry a length of zero and fall out of both kernels.
  int numSlots = numTokens / kChunk + numSeq;
  int headRatio = numVHead / numKHead;

  Tensor chunkStart = createCudaTensorInt32({numSlots});
  Tensor chunkLen = createCudaTensorInt32({numSlots});
  Tensor seqChunkBegin = createCudaTensorInt32({numSeq});
  Tensor seqChunkCount = createCudaTensorInt32({numSeq});
  Tensor a = createCudaTensorFloat({numSlots, numVHead, kChunk, kChunk});
  Tensor rhs = createCudaTensorHalf({numSlots, numVHead, kChunk, 2 * headDim});
  Tensor mmat = createCudaTensorHalf({numSlots, numVHead, kChunk, kChunk});
  Tensor cexp = createCudaTensorFloat({numSlots, numVHead, kChunk});
  Tensor clast = createCudaTensorFloat({numSlots, numVHead, kChunk});

  chunkTableKernel<<<1, kTableThreads>>>(
      getDataPtrCuda<int32_t>(cuSeqlens),
      getDataPtrCuda<int32_t>(chunkStart),
      getDataPtrCuda<int32_t>(chunkLen),
      getDataPtrCuda<int32_t>(seqChunkBegin),
      getDataPtrCuda<int32_t>(seqChunkCount),
      numSeq,
      numSlots);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  buildChunkKernel<<<dim3(numSlots, numVHead), kBuildThreads>>>(
      getDataPtrCuda<half>(q),
      getDataPtrCuda<half>(k),
      getDataPtrCuda<half>(v),
      getDataPtrCuda<float>(g),
      getDataPtrCuda<float>(beta),
      getDataPtrCuda<int32_t>(chunkStart),
      getDataPtrCuda<int32_t>(chunkLen),
      getDataPtrCuda<float>(a),
      getDataPtrCuda<half>(rhs),
      getDataPtrCuda<half>(mmat),
      getDataPtrCuda<float>(cexp),
      getDataPtrCuda<float>(clast),
      numKHead,
      numVHead,
      headDim,
      headRatio);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  // Every chunk of every head at once: (I + A) [U | W] = [beta V | diag(beta exp(c)) K].
  triangularSolveInplace(a, rhs);

  // sState, then u, then the two staged tiles. The tiles carry the odd row padding, and the wider
  // of the two ways sH is indexed -- kChunk by kTileStride -- is what sizes it.
  size_t smem = static_cast<size_t>(headDim * kValueBlock) * sizeof(float) +
                static_cast<size_t>(kChunk * kValueBlock + 2 * kChunk * kTileStride) *
                    sizeof(half);
  LL_CHECK_CUDA_STATUS(cudaFuncSetAttribute(
      scanKernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem)));

  dim3 scanGrid(numSeq, numVHead * (headDim / kValueBlock));
  scanKernel<<<scanGrid, kScanThreads, smem>>>(
      getDataPtrCuda<half>(q),
      getDataPtrCuda<half>(k),
      getDataPtrCuda<half>(rhs),
      getDataPtrCuda<half>(mmat),
      getDataPtrCuda<float>(cexp),
      getDataPtrCuda<float>(clast),
      getDataPtrCuda<int32_t>(chunkStart),
      getDataPtrCuda<int32_t>(chunkLen),
      getDataPtrCuda<int32_t>(seqChunkBegin),
      getDataPtrCuda<int32_t>(seqChunkCount),
      getDataPtrCuda<int32_t>(stateSlots),
      getDataPtrCuda<float>(state),
      getDataPtrCuda<half>(o),
      numKHead,
      numVHead,
      headDim,
      headRatio);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace chunked

/// The single-launch implementation: one CTA owns a (sequence, value head) from its incoming state
/// to its outgoing one, and builds, solves and scans every chunk without going back to global
/// memory for anything but q, k, v and the output.
namespace fused {
namespace {

/// The chunk length. Shorter than the chunked path's, and deliberately: the decomposition is exact
/// at any chunk length, and the work a chunk costs splits into a part that scales with D squared
/// and a part that scales with C times D, so a shorter chunk is strictly less arithmetic. What
/// stops it going shorter still is that every chunk is one step of a sequential scan.
constexpr int kChunk = 32;

/// Every phase lays its output over the same grid of row groups by column groups, which is what
/// lets an accumulator computed in one phase stay in the registers of the thread that finishes it
/// in another. The state fills so much shared memory that only one CTA is ever resident per SM, so
/// the CTA has to be wide enough on its own to keep that SM busy.
constexpr int kRowGroups = 16;
constexpr int kColGroups = 16;
constexpr int kThreads = kRowGroups * kColGroups;

/// A kChunk by kChunk output, one thread's rows by its columns.
constexpr int kTileA = kChunk / kRowGroups;
constexpr int kTileB = kChunk / kColGroups;

/// Rows of a kChunk-tall output one thread owns, and value columns one thread owns.
constexpr int kTileRow = kChunk / kRowGroups;
constexpr int kTileCol = 8;

/// Rows of a D-tall output one thread owns, which is what caps the head dimension: the state has
/// to be covered by one pass of the thread grid, and to fit in shared memory besides.
constexpr int kTileState = 8;
constexpr int kMaxHeadDim = kRowGroups * kTileState;

/// Column padding on the transposed key and query buffers. They are read down a column by threads
/// whose rows are two apart, and at an even stride those rows all land in the same bank.
constexpr int kStride = kChunk + 1;

/// Shared memory one CTA needs for a head dimension of `d`. The state is nearly all of it.
size_t sharedBytes(int d) {
  return static_cast<size_t>(d) * d * sizeof(float) +               // sState
         static_cast<size_t>(kChunk) * kStride * sizeof(float) +    // sA
         static_cast<size_t>(kChunk) * kStride * sizeof(half) +     // sM
         static_cast<size_t>(d) * kStride * sizeof(half) * 2 +      // sKT, sQT
         static_cast<size_t>(kChunk) * d * sizeof(half) +           // sB
         static_cast<size_t>(4 * kChunk) * sizeof(float);           // the decays
}

}  // namespace

/// Fusing is not only about traffic. Once the state is in the CTA the chunk's triangular system can
/// be solved against the right hand side it actually has,
///
///   (I + A) u = beta * V - diag(beta exp(c)) K S_0,   A_ij = beta_i exp(c_i - c_j) k_i . k_j
///
/// rather than against the two state-free ones a separate solve is forced to use. The chunked path
/// has to solve for both U and W and then spend a C by D by D product on u = U - W S_0; this solves
/// half as wide and folds that product into the right hand side it was already building.
///
/// The phases, per chunk, all over the same 16 by 16 thread grid:
///
///   1. the cumulative log decay of the chunk
///   2. A and the decayed score matrix M, from K K^T and Q K^T
///   3. K S_0 and Q S_0, the only two passes over the state that read it
///   4. the right hand side, from K S_0
///   5. forward substitution, one thread per value column, entirely in shared memory
///   6. the output, exp(c) Q S_0 + M u, from an accumulator phase 3 left in registers
///   7. the state, exp(c_last) S_0 + K~^T u
__global__ __launch_bounds__(kThreads) void gatedDeltaNetFusedKernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    const float *__restrict__ g,
    const float *__restrict__ beta,
    const int32_t *__restrict__ cuSeqlens,
    const int32_t *__restrict__ stateSlots,
    float *__restrict__ state,
    half *__restrict__ out,
    int numKHead,
    int numVHead,
    int headDim,
    int headRatio) {
  extern __shared__ char smem[];
  const int d = headDim;

  float *sState = reinterpret_cast<float *>(smem);
  float *sA = sState + static_cast<int64_t>(d) * d;
  half *sM = reinterpret_cast<half *>(sA + kChunk * kStride);
  half *sKT = sM + kChunk * kStride;
  half *sQT = sKT + static_cast<int64_t>(d) * kStride;
  half *sB = sQT + static_cast<int64_t>(d) * kStride;
  float *sCum = reinterpret_cast<float *>(sB + static_cast<int64_t>(kChunk) * d);
  float *sCexp = sCum + kChunk;
  float *sClast = sCexp + kChunk;
  float *sBeta = sClast + kChunk;

  const int head = blockIdx.x;
  const int seq = blockIdx.y;
  const int kHead = head / headRatio;
  const int tid = threadIdx.x;
  const int ty = tid / kColGroups;
  const int tx = tid - ty * kColGroups;
  const int col0 = tx * kTileCol;
  const bool colInRange = col0 < d;

  const int64_t stateOffset = (static_cast<int64_t>(stateSlots[seq]) * numVHead + head) * d * d;
  for (int idx = tid; idx < d * d; idx += kThreads) {
    sState[idx] = state[stateOffset + idx];
  }

  const int begin = cuSeqlens[seq];
  const int end = cuSeqlens[seq + 1];

  for (int t0 = begin; t0 < end; t0 += kChunk) {
    const int len = min(kChunk, end - t0);
    __syncthreads();

    // -- 1. the cumulative log decay, kept in the log domain so that every decay below is exp of a
    // difference rather than a ratio of two exponentials that may both have underflowed.
    // An inclusive scan across one warp, the chunk being exactly a warp long. Walking it on a
    // single thread instead costs 32 dependent steps over 32 serial global loads, and with one CTA
    // per SM the other seven warps spend all of it waiting on the barrier below.
    if (tid < kChunk) {
      float x = (tid < len) ? g[static_cast<int64_t>(t0 + tid) * numVHead + head] : 0.0f;
#pragma unroll
      for (int offset = 1; offset < kChunk; offset <<= 1) {
        float y = __shfl_up_sync(0xffffffff, x, offset);
        if (tid >= offset) x += y;
      }
      sCum[tid] = x;
    }
    __syncthreads();

    const float cumLast = sCum[len - 1];
    for (int i = tid; i < len; i += kThreads) {
      sCexp[i] = expf(sCum[i]);
      sClast[i] = expf(cumLast - sCum[i]);
      sBeta[i] = beta[static_cast<int64_t>(t0 + i) * numVHead + head];
    }

    // The keys and queries of the chunk, transposed. Two of the three phases that read them walk
    // the head dimension as the reduction and the third walks it as the output, and this is the
    // layout that keeps all three off the same shared memory bank.
    for (int idx = tid; idx < kChunk * d; idx += kThreads) {
      int j = idx / d;
      int m = idx - j * d;
      bool ok = j < len;
      int64_t off = (static_cast<int64_t>(t0 + j) * numKHead + kHead) * d + m;
      sKT[m * kStride + j] = ok ? k[off] : __float2half(0.0f);
      sQT[m * kStride + j] = ok ? q[off] : __float2half(0.0f);
    }
    __syncthreads();

    // -- 2. A and M, from one pass over the head dimension.
    float accA[kTileA][kTileB] = {};
    float accM[kTileA][kTileB] = {};
    for (int c = 0; c < d; ++c) {
      const half *kRow = sKT + c * kStride;
      const half *qRow = sQT + c * kStride;

      float kj[kTileB];
#pragma unroll
      for (int b = 0; b < kTileB; ++b) kj[b] = __half2float(kRow[tx * kTileB + b]);

#pragma unroll
      for (int a = 0; a < kTileA; ++a) {
        float ki = __half2float(kRow[ty * kTileA + a]);
        float qi = __half2float(qRow[ty * kTileA + a]);
#pragma unroll
        for (int b = 0; b < kTileB; ++b) {
          accA[a][b] += ki * kj[b];
          accM[a][b] += qi * kj[b];
        }
      }
    }

#pragma unroll
    for (int a = 0; a < kTileA; ++a) {
      int i = ty * kTileA + a;
#pragma unroll
      for (int b = 0; b < kTileB; ++b) {
        int j = tx * kTileB + b;
        float av = 0.0f;
        float mv = 0.0f;
        if (i < len && j <= i) {
          float decay = expf(sCum[i] - sCum[j]);
          if (j < i) av = sBeta[i] * decay * accA[a][b];
          mv = decay * accM[a][b];
        }

        // The identity part of (I + A), for the rows a short chunk does not use as much as for the
        // ones it does, so the substitution leaves those rows alone.
        if (i == j) av = 1.0f;
        sA[i * kStride + j] = av;
        sM[i * kStride + j] = __float2half(mv);
      }
    }

    // -- 3. K S_0 and Q S_0. Both reduce over the same axis against the same state, so they share
    // one pass over it and one set of loads.
    float accK[kTileRow][kTileCol] = {};
    float accQ[kTileRow][kTileCol] = {};
    if (colInRange) {
      for (int m = 0; m < d; ++m) {
        const float *sRow = sState + static_cast<int64_t>(m) * d + col0;
        float4 lo = *reinterpret_cast<const float4 *>(sRow);
        float4 hi = *reinterpret_cast<const float4 *>(sRow + 4);
        float sv[kTileCol] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

        const half *kRow = sKT + m * kStride;
        const half *qRow = sQT + m * kStride;
#pragma unroll
        for (int a = 0; a < kTileRow; ++a) {
          float kv = __half2float(kRow[ty * kTileRow + a]);
          float qv = __half2float(qRow[ty * kTileRow + a]);
#pragma unroll
          for (int b = 0; b < kTileCol; ++b) {
            accK[a][b] += kv * sv[b];
            accQ[a][b] += qv * sv[b];
          }
        }
      }
    }

    // -- 4. the right hand side, beta_i (v_i - exp(c_i) (K S_0)_i).
    if (colInRange) {
#pragma unroll
      for (int a = 0; a < kTileRow; ++a) {
        int i = ty * kTileRow + a;
        if (i >= len) {
#pragma unroll
          for (int b = 0; b < kTileCol; ++b) sB[i * d + col0 + b] = __float2half(0.0f);
          continue;
        }

        const half *vRow = v + (static_cast<int64_t>(t0 + i) * numVHead + head) * d + col0;
        float scale = sBeta[i];
        float decay = sCexp[i];
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) {
          sB[i * d + col0 + b] =
              __float2half(scale * (__half2float(vRow[b]) - decay * accK[a][b]));
        }
      }
    }
    __syncthreads();

    // -- 5. forward substitution. Every column is an independent system and each thread owns one
    // outright, so the rows it reads back are rows it wrote itself and this needs no barrier.
    if (tid < d) {
      for (int i = 0; i < len; ++i) {
        const float *aRow = sA + i * kStride;
        float acc = __half2float(sB[i * d + tid]);
        for (int j = 0; j < i; ++j) {
          acc -= aRow[j] * __half2float(sB[j * d + tid]);
        }
        sB[i * d + tid] = __float2half(acc / aRow[i]);
      }
    }
    __syncthreads();

    // -- 6. the output. accQ has been sitting in registers since phase 3, in the thread that owns
    // this very tile of it.
    if (colInRange) {
      float accO[kTileRow][kTileCol];
#pragma unroll
      for (int a = 0; a < kTileRow; ++a) {
        int i = ty * kTileRow + a;
        float scale = (i < len) ? sCexp[i] : 0.0f;
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) accO[a][b] = scale * accQ[a][b];
      }

      // M is already zero above its diagonal and outside the chunk, so this needs no mask.
      for (int j = 0; j < len; ++j) {
        float uv[kTileCol];
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) uv[b] = __half2float(sB[j * d + col0 + b]);

#pragma unroll
        for (int a = 0; a < kTileRow; ++a) {
          float mv = __half2float(sM[(ty * kTileRow + a) * kStride + j]);
#pragma unroll
          for (int b = 0; b < kTileCol; ++b) accO[a][b] += mv * uv[b];
        }
      }

#pragma unroll
      for (int a = 0; a < kTileRow; ++a) {
        int i = ty * kTileRow + a;
        if (i >= len) continue;

        half *oRow = out + (static_cast<int64_t>(t0 + i) * numVHead + head) * d + col0;
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) oRow[b] = __float2half(accO[a][b]);
      }
    }

    // -- 7. the state. Each thread reads and writes only its own tile of it, so the read at the top
    // and the write at the bottom race with nobody; the barrier after phase 4 is what separates
    // this from phase 3's reads of the state it is about to replace.
    const float gamma = sCexp[len - 1];
    const int row0 = ty * kTileState;
    if (colInRange && row0 < d) {
      float accS[kTileState][kTileCol];
#pragma unroll
      for (int a = 0; a < kTileState; ++a) {
        int m = row0 + a;
        if (m < d) {
          const float *sRow = sState + static_cast<int64_t>(m) * d + col0;
          float4 lo = *reinterpret_cast<const float4 *>(sRow);
          float4 hi = *reinterpret_cast<const float4 *>(sRow + 4);
          accS[a][0] = gamma * lo.x;
          accS[a][1] = gamma * lo.y;
          accS[a][2] = gamma * lo.z;
          accS[a][3] = gamma * lo.w;
          accS[a][4] = gamma * hi.x;
          accS[a][5] = gamma * hi.y;
          accS[a][6] = gamma * hi.z;
          accS[a][7] = gamma * hi.w;
        } else {
#pragma unroll
          for (int b = 0; b < kTileCol; ++b) accS[a][b] = 0.0f;
        }
      }

      for (int j = 0; j < len; ++j) {
        float uv[kTileCol];
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) uv[b] = __half2float(sB[j * d + col0 + b]);

        float decay = sClast[j];
#pragma unroll
        for (int a = 0; a < kTileState; ++a) {
          if (row0 + a >= d) continue;

          float kv = decay * __half2float(sKT[(row0 + a) * kStride + j]);
#pragma unroll
          for (int b = 0; b < kTileCol; ++b) accS[a][b] += kv * uv[b];
        }
      }

#pragma unroll
      for (int a = 0; a < kTileState; ++a) {
        int m = row0 + a;
        if (m >= d) continue;

        float *sRow = sState + static_cast<int64_t>(m) * d + col0;
        *reinterpret_cast<float4 *>(sRow) =
            make_float4(accS[a][0], accS[a][1], accS[a][2], accS[a][3]);
        *reinterpret_cast<float4 *>(sRow + 4) =
            make_float4(accS[a][4], accS[a][5], accS[a][6], accS[a][7]);
      }
    }
  }

  __syncthreads();
  for (int idx = tid; idx < d * d; idx += kThreads) {
    state[stateOffset + idx] = sState[idx];
  }
}

/// Whether this device can run the fused kernel at this head dimension at all.
bool fits(int headDim, int *smemOut) {
  if (headDim % kTileCol != 0 || headDim > kMaxHeadDim) return false;

  size_t smem = sharedBytes(headDim);
  int maxSmem = 0;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(
      &maxSmem,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      0));
  if (smem > static_cast<size_t>(maxSmem)) return false;

  *smemOut = static_cast<int>(smem);
  return true;
}

void run(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &g,
    const Tensor &beta,
    const Tensor &cuSeqlens,
    const Tensor &stateSlots,
    Tensor &state,
    Tensor &o,
    int numKHead,
    int numVHead,
    int headDim,
    int numSeq) {
  int smem = 0;
  CHECK(fits(headDim, &smem))
      << "the fused gatedDeltaNetPrefill needs a head dimension that is a multiple of " << kTileCol
      << ", at most " << kMaxHeadDim << ", and whose state fits in this device's shared memory; "
      << "got " << headDim;

  LL_CHECK_CUDA_STATUS(cudaFuncSetAttribute(
      gatedDeltaNetFusedKernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem));

  gatedDeltaNetFusedKernel<<<dim3(numVHead, numSeq), kThreads, smem>>>(
      getDataPtrCuda<half>(q),
      getDataPtrCuda<half>(k),
      getDataPtrCuda<half>(v),
      getDataPtrCuda<float>(g),
      getDataPtrCuda<float>(beta),
      getDataPtrCuda<int32_t>(cuSeqlens),
      getDataPtrCuda<int32_t>(stateSlots),
      getDataPtrCuda<float>(state),
      getDataPtrCuda<half>(o),
      numKHead,
      numVHead,
      headDim,
      numVHead / numKHead);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace fused

/// kFused with the state in registers instead of shared memory.
///
/// The state is what forced the fused kernel to one CTA per SM: 64 KB of the 99 KB a block may
/// have. Spreading it over the CTA's registers instead -- 16384 floats over 256 threads is 64 each
/// -- leaves shared memory holding only the chunk, and two CTAs fit. The cost is that the state is
/// now partitioned across threads, and two of the phases reduce along the axis it is partitioned
/// on: a thread can only form a partial sum of K S_0 and Q S_0, and those partials have to be
/// summed across the threads that split the rows.
///
/// Which is why the row groups are the low bits of the thread index. A butterfly over them then
/// stays inside one warp and costs shuffles rather than another trip through shared memory, and
/// every lane ends up holding the finished sum.
namespace fusedreg {
namespace {

constexpr int kChunk = 32;

/// 512 threads was tried, to halve the state a thread holds and so its register pressure. It is
/// worse: two CTAs per SM then needs 64 registers a thread, which is tighter than the 128 the same
/// budget allows at 256, and the compiler spills three times as much onto the stack.
constexpr int kThreads = 256;

/// The state tile one thread owns: kTileM rows by kTileN columns, which is the whole state divided
/// by the CTA. Wider rows make the reduction cheaper -- fewer thread groups to sum across -- and
/// wider columns make the pass over the state cheaper, since one key element then feeds more
/// multiply-adds.
constexpr int kTileM = 16;
constexpr int kTileN = 4;

/// Row padding on the staged key and query tiles, in elements. Two elements keeps the row stride
/// odd in words, which is what the A and M build wants, while staying 4-byte aligned so the passes
/// that walk a row can read it as half2.
constexpr int kRowPad = 2;

/// The output tiles of the phases that are laid out over a 16 by 16 grid rather than over the
/// state: a kChunk by D output, kTileRow rows by kTileCol columns each.
constexpr int kGroups = 16;
constexpr int kTileRow = kChunk / (kThreads / kGroups);
constexpr int kTileCol = 8;

/// The column tile of the kChunk by kChunk matrices, which are square where the others are wide.
constexpr int kTileSquare = kChunk / kGroups;

constexpr int kMaxHeadDim = 128;

size_t sharedBytes(int d) {
  return static_cast<size_t>(kChunk) * (kChunk + 1) * sizeof(float) +   // sA
         static_cast<size_t>(kChunk) * (kChunk + 1) * sizeof(half) +    // sM
         static_cast<size_t>(kChunk) * (d + kRowPad) * sizeof(half) * 2 +  // sK, sQ
         static_cast<size_t>(kChunk) * d * sizeof(half) * 2 +           // sB, sQS
         static_cast<size_t>(4 * kChunk) * sizeof(float);               // the decays
}

}  // namespace

__global__ __launch_bounds__(kThreads, 2) void gatedDeltaNetFusedRegKernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    const float *__restrict__ g,
    const float *__restrict__ beta,
    const int32_t *__restrict__ cuSeqlens,
    const int32_t *__restrict__ stateSlots,
    float *__restrict__ state,
    half *__restrict__ out,
    int numKHead,
    int numVHead,
    int headDim,
    int headRatio) {
  extern __shared__ char smem[];
  const int d = headDim;
  const int rowStride = d + kRowPad;

  float *sA = reinterpret_cast<float *>(smem);
  half *sM = reinterpret_cast<half *>(sA + kChunk * (kChunk + 1));
  half *sK = sM + kChunk * (kChunk + 1);
  half *sQ = sK + static_cast<int64_t>(kChunk) * rowStride;
  half *sB = sQ + static_cast<int64_t>(kChunk) * rowStride;
  half *sQS = sB + static_cast<int64_t>(kChunk) * d;
  float *sCum = reinterpret_cast<float *>(sQS + static_cast<int64_t>(kChunk) * d);
  float *sCexp = sCum + kChunk;
  float *sClast = sCexp + kChunk;
  float *sBeta = sClast + kChunk;

  const int head = blockIdx.x;
  const int seq = blockIdx.y;
  const int kHead = head / headRatio;
  const int tid = threadIdx.x;

  // The row groups are the low bits so that the butterfly below stays inside a warp.
  const int numRowGroups = d / kTileM;
  const int rowGroup = tid & (numRowGroups - 1);
  const int colGroup = tid / numRowGroups;
  const int row0 = rowGroup * kTileM;
  const int col0 = colGroup * kTileN;
  const bool ownsState = col0 < d;

  // The 16 by 16 grid the phases that do not touch the state are laid out over.
  const int ty = tid / kGroups;
  const int tx = tid - ty * kGroups;
  const int wide0 = tx * kTileCol;
  const bool wideInRange = wide0 < d;

  const int64_t stateOffset = (static_cast<int64_t>(stateSlots[seq]) * numVHead + head) * d * d;
  float accState[kTileM][kTileN];
#pragma unroll
  for (int a = 0; a < kTileM; ++a) {
#pragma unroll
    for (int b = 0; b < kTileN; ++b) {
      accState[a][b] = ownsState
                           ? state[stateOffset + static_cast<int64_t>(row0 + a) * d + col0 + b]
                           : 0.0f;
    }
  }

  const int begin = cuSeqlens[seq];
  const int end = cuSeqlens[seq + 1];

  for (int t0 = begin; t0 < end; t0 += kChunk) {
    const int len = min(kChunk, end - t0);
    __syncthreads();

    if (tid < kChunk) {
      float x = (tid < len) ? g[static_cast<int64_t>(t0 + tid) * numVHead + head] : 0.0f;
#pragma unroll
      for (int offset = 1; offset < kChunk; offset <<= 1) {
        float y = __shfl_up_sync(0xffffffff, x, offset);
        if (tid >= offset) x += y;
      }
      sCum[tid] = x;
    }
    __syncthreads();

    const float cumLast = sCum[len - 1];
    for (int i = tid; i < len; i += kThreads) {
      sCexp[i] = expf(sCum[i]);
      sClast[i] = expf(cumLast - sCum[i]);
      sBeta[i] = beta[static_cast<int64_t>(t0 + i) * numVHead + head];
    }

    // The chunk's keys and queries, row major this time: every phase that reads them here walks a
    // row, either as the reduction or as the state rows the thread owns.
    for (int idx = tid; idx < kChunk * d; idx += kThreads) {
      int j = idx / d;
      int m = idx - j * d;
      bool ok = j < len;
      int64_t off = (static_cast<int64_t>(t0 + j) * numKHead + kHead) * d + m;
      sK[j * rowStride + m] = ok ? k[off] : __float2half(0.0f);
      sQ[j * rowStride + m] = ok ? q[off] : __float2half(0.0f);
    }
    __syncthreads();

    // -- A and M.
    {
      float accA[kTileRow][kTileSquare] = {};
      float accM[kTileRow][kTileSquare] = {};
      const int ai = ty * kTileRow;
      const int aj = tx * kTileSquare;
      for (int c = 0; c < d; ++c) {
        float kj[kTileSquare];
#pragma unroll
        for (int b = 0; b < kTileSquare; ++b) {
          kj[b] = __half2float(sK[(aj + b) * rowStride + c]);
        }

#pragma unroll
        for (int a = 0; a < kTileRow; ++a) {
          float ki = __half2float(sK[(ai + a) * rowStride + c]);
          float qi = __half2float(sQ[(ai + a) * rowStride + c]);
#pragma unroll
          for (int b = 0; b < kTileSquare; ++b) {
            accA[a][b] += ki * kj[b];
            accM[a][b] += qi * kj[b];
          }
        }
      }

#pragma unroll
      for (int a = 0; a < kTileRow; ++a) {
        int i = ai + a;
#pragma unroll
      for (int b = 0; b < kTileSquare; ++b) {
          int j = aj + b;
          float av = 0.0f;
          float mv = 0.0f;
          if (i < len && j <= i) {
            float decay = expf(sCum[i] - sCum[j]);
            if (j < i) av = sBeta[i] * decay * accA[a][b];
            mv = decay * accM[a][b];
          }
          if (i == j) av = 1.0f;
          sA[i * (kChunk + 1) + j] = av;
          sM[i * (kChunk + 1) + j] = __float2half(mv);
        }
      }
    }

    // -- K S_0 and Q S_0, one chunk row at a time. Every thread runs this, including the ones whose
    // state tile is outside a small head dimension: their tile is zero and the shuffles below need
    // the whole warp.
    for (int i = 0; i < len; ++i) {
      const half *kRow = sK + i * rowStride + row0;
      const half *qRow = sQ + i * rowStride + row0;

      float pk[kTileN] = {};
      float pq[kTileN] = {};
#pragma unroll
      for (int a = 0; a < kTileM; a += 2) {
        float2 kv = __half22float2(*reinterpret_cast<const half2 *>(kRow + a));
        float2 qv = __half22float2(*reinterpret_cast<const half2 *>(qRow + a));
#pragma unroll
        for (int b = 0; b < kTileN; ++b) {
          pk[b] += kv.x * accState[a][b] + kv.y * accState[a + 1][b];
          pq[b] += qv.x * accState[a][b] + qv.y * accState[a + 1][b];
        }
      }

      // The partials of the threads that split these rows, summed across them. Every lane keeps the
      // finished sum, which is what lets any of them do the store below.
      for (int offset = 1; offset < numRowGroups; offset <<= 1) {
#pragma unroll
        for (int b = 0; b < kTileN; ++b) {
          pk[b] += __shfl_xor_sync(0xffffffff, pk[b], offset);
          pq[b] += __shfl_xor_sync(0xffffffff, pq[b], offset);
        }
      }

      if (rowGroup == 0 && ownsState) {
        const half *vRow = v + (static_cast<int64_t>(t0 + i) * numVHead + head) * d + col0;
        float scale = sBeta[i];
        float decay = sCexp[i];
#pragma unroll
        for (int b = 0; b < kTileN; ++b) {
          sB[i * d + col0 + b] =
              __float2half(scale * (__half2float(vRow[b]) - decay * pk[b]));
          sQS[i * d + col0 + b] = __float2half(pq[b]);
        }
      }
    }

    __syncthreads();

    // -- forward substitution, one thread per value column.
    if (tid < d) {
      for (int i = 0; i < len; ++i) {
        const float *aRow = sA + i * (kChunk + 1);
        float acc = __half2float(sB[i * d + tid]);
        for (int j = 0; j < i; ++j) {
          acc -= aRow[j] * __half2float(sB[j * d + tid]);
        }
        sB[i * d + tid] = __float2half(acc / aRow[i]);
      }
    }
    __syncthreads();

    // -- the output.
    if (wideInRange) {
      float accO[kTileRow][kTileCol];
#pragma unroll
      for (int a = 0; a < kTileRow; ++a) {
        int i = ty * kTileRow + a;
        float scale = (i < len) ? sCexp[i] : 0.0f;
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) {
          accO[a][b] = scale * __half2float(sQS[i * d + wide0 + b]);
        }
      }

      for (int j = 0; j < len; ++j) {
        float uv[kTileCol];
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) uv[b] = __half2float(sB[j * d + wide0 + b]);

#pragma unroll
        for (int a = 0; a < kTileRow; ++a) {
          float mv = __half2float(sM[(ty * kTileRow + a) * (kChunk + 1) + j]);
#pragma unroll
          for (int b = 0; b < kTileCol; ++b) accO[a][b] += mv * uv[b];
        }
      }

#pragma unroll
      for (int a = 0; a < kTileRow; ++a) {
        int i = ty * kTileRow + a;
        if (i >= len) continue;

        half *oRow = out + (static_cast<int64_t>(t0 + i) * numVHead + head) * d + wide0;
#pragma unroll
        for (int b = 0; b < kTileCol; ++b) oRow[b] = __float2half(accO[a][b]);
      }
    }

    // -- the state, which needs no reduction: the rows are the output here, not the sum.
    if (ownsState) {
      const float gamma = sCexp[len - 1];
#pragma unroll
      for (int a = 0; a < kTileM; ++a) {
#pragma unroll
        for (int b = 0; b < kTileN; ++b) accState[a][b] *= gamma;
      }

      for (int j = 0; j < len; ++j) {
        float uv[kTileN];
#pragma unroll
        for (int b = 0; b < kTileN; ++b) uv[b] = __half2float(sB[j * d + col0 + b]);

        const half *kRow = sK + j * rowStride + row0;
        float decay = sClast[j];
#pragma unroll
        for (int a = 0; a < kTileM; a += 2) {
          float2 kv = __half22float2(*reinterpret_cast<const half2 *>(kRow + a));
          float k0 = decay * kv.x;
          float k1 = decay * kv.y;
#pragma unroll
          for (int b = 0; b < kTileN; ++b) {
            accState[a][b] += k0 * uv[b];
            accState[a + 1][b] += k1 * uv[b];
          }
        }
      }
    }
  }

  if (ownsState) {
#pragma unroll
    for (int a = 0; a < kTileM; ++a) {
#pragma unroll
      for (int b = 0; b < kTileN; ++b) {
        state[stateOffset + static_cast<int64_t>(row0 + a) * d + col0 + b] = accState[a][b];
      }
    }
  }
}

bool fits(int headDim, int *smemOut) {
  // A multiple of the state tile in both directions, of the wide tile the output phase uses, and
  // no more row groups than a warp can reduce across.
  if (headDim % kTileM != 0 || headDim % kTileN != 0 || headDim % kTileCol != 0) return false;
  if (headDim > kMaxHeadDim || headDim / kTileM > 32) return false;

  size_t smem = sharedBytes(headDim);
  int maxSmem = 0;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(
      &maxSmem,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      0));
  if (smem > static_cast<size_t>(maxSmem)) return false;

  *smemOut = static_cast<int>(smem);
  return true;
}

void run(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &g,
    const Tensor &beta,
    const Tensor &cuSeqlens,
    const Tensor &stateSlots,
    Tensor &state,
    Tensor &o,
    int numKHead,
    int numVHead,
    int headDim,
    int numSeq) {
  int smem = 0;
  CHECK(fits(headDim, &smem))
      << "the register-state gatedDeltaNetPrefill needs a head dimension that is a multiple of "
      << kTileM << " and at most " << kMaxHeadDim << ", got " << headDim;

  LL_CHECK_CUDA_STATUS(cudaFuncSetAttribute(
      gatedDeltaNetFusedRegKernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem));

  gatedDeltaNetFusedRegKernel<<<dim3(numVHead, numSeq), kThreads, smem>>>(
      getDataPtrCuda<half>(q),
      getDataPtrCuda<half>(k),
      getDataPtrCuda<half>(v),
      getDataPtrCuda<float>(g),
      getDataPtrCuda<float>(beta),
      getDataPtrCuda<int32_t>(cuSeqlens),
      getDataPtrCuda<int32_t>(stateSlots),
      getDataPtrCuda<float>(state),
      getDataPtrCuda<half>(o),
      numKHead,
      numVHead,
      headDim,
      numVHead / numKHead);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace fusedreg

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

  Tensor o = createCudaTensorHalf({numTokens, numVHead, headDim});
  if (numSeq == 0 || numTokens == 0) return o;

  // Nothing the FP32 paths do is faster than the tensor core one where it runs, so kAuto stops
  // here: the choice below it is between three implementations of the same arithmetic, and only
  // matters for the head dimensions this one cannot take.
  int tcSmem = 0;
  if ((path == GatedDeltaNetPath::kAuto && gdnmma::fits(headDim, &tcSmem)) ||
      path == GatedDeltaNetPath::kTensorCoreMma) {
    gdnmma::run(
        q, k, v, g, beta, cuSeqlens, stateSlots, state, o, numKHead, numVHead, headDim, numSeq);

    LL_CUDA_SYNCHRONIZE();
    LL_CHECK_CUDA_STATUS(cudaGetLastError());
    return o;
  }

  if ((path == GatedDeltaNetPath::kAuto && gdnwmma::fits(headDim, &tcSmem)) ||
      path == GatedDeltaNetPath::kTensorCore) {
    gdnwmma::run(
        q, k, v, g, beta, cuSeqlens, stateSlots, state, o, numKHead, numVHead, headDim, numSeq);

    LL_CUDA_SYNCHRONIZE();
    LL_CHECK_CUDA_STATUS(cudaGetLastError());
    return o;
  }

  int smem = 0;
  bool useFused = fused::fits(headDim, &smem);
  if (path == GatedDeltaNetPath::kChunked || path == GatedDeltaNetPath::kFusedRegisters) {
    useFused = false;
  } else if (path == GatedDeltaNetPath::kAuto && useFused) {
    // The fused kernel gets one CTA per (sequence, value head) and only one of them is resident per
    // SM, so what decides it is how well that CTA count packs the device rather than how big the
    // prefill is. A last wave that leaves more than a fifth of the SMs idle costs more than the
    // traffic the fusion saves; measured, the turn is between one sequence and two.
    int numSm = 0;
    LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, 0));
    int numCta = numSeq * numVHead;
    int waves = (numCta + numSm - 1) / numSm;
    useFused = numCta * 5 >= waves * numSm * 4;
  }

  if (path == GatedDeltaNetPath::kFusedRegisters) {
    fusedreg::run(
        q, k, v, g, beta, cuSeqlens, stateSlots, state, o, numKHead, numVHead, headDim, numSeq);
  } else if (useFused) {
    fused::run(
        q, k, v, g, beta, cuSeqlens, stateSlots, state, o, numKHead, numVHead, headDim, numSeq);
  } else {
    chunked::run(
        q,
        k,
        v,
        g,
        beta,
        cuSeqlens,
        stateSlots,
        state,
        o,
        numTokens,
        numKHead,
        numVHead,
        headDim,
        numSeq);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return o;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
