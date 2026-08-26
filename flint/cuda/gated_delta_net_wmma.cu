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

#include "flint/cuda/gated_delta_net_wmma.h"

#include <cuda_fp16.h>
#include <mma.h>

#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {
namespace gdnwmma {

using namespace nvcuda;

/// The tensor core prefill. One CTA owns a (sequence, value head) from its incoming state to its
/// outgoing one, like the other fused paths, but it is arranged so that every product larger than a
/// vector is an HMMA rather than a loop of FP32 fused multiply-adds.
///
/// Three things had to change for that, and each of them is why a phase below looks the way it
/// does:
///
///   1. **The state is an accumulator.** It lives in the CTA's registers as WMMA accumulator
///      fragments -- (D, D) floats over 512 threads is 32 a thread -- because the last thing a
///      chunk does, S <- exp(c_last) S + K~^T u, is a GEMM whose C operand is the state itself. An
///      accumulator's element order is opaque, so the two phases that *read* the state as an
///      operand cannot read the registers: the chunk starts by writing the state out to shared
///      memory as half, once, and both passes read it from there.
///   2. **The system is inverted, not substituted.** Forward substitution is inherently serial down
///      the rows of a chunk and cannot be a tensor core instruction. Instead (I + A)^-1 is formed
///      explicitly -- four 16 by 16 diagonal blocks by Gauss-Jordan, then two blocked combines that
///      are themselves HMMAs -- and the chunk's u is a GEMM against it. The inverse costs about 3%
///      of the chunk's multiply-adds and moves the other 97% onto the tensor cores. It also belongs
///      to four warps alone, which the other twelve do not wait for: nothing in phase 5 reads it.
///   3. **The epilogues stay in registers.** Every phase that has to know which (row, column) an
///      accumulator element is -- the decays on the score matrices, the right hand side's beta and
///      v, the output -- would have to store the accumulator to shared memory and read it back by
///      coordinate, since the WMMA API does not say where an element lives. It does not: the warp
///      probes the layout once at launch, by loading a tile that holds each element's own index,
///      and every epilogue after that reads its own registers. Probing is not a guess about the
///      hardware -- whatever comes back is what this device does.
///
/// The phases, per chunk of 64 tokens:
///
///   0. the cumulative log decay, the only serial scan left, on one warp
///   1. K and Q into shared memory by cp.async, rows past the chunk's length zeroed
///   2. the state into shared memory as half, out of the accumulator fragments
///   3. A = K K^T and M = Q K^T, decayed and masked into (I + A) and the score matrix
///   4. (I + A)^-1, blockwise, on the first four warps
///   5. K S_0 and Q S_0, the two passes over the state, in one pass over its operand fragments
///   6. the right hand side, beta (v - exp(c) K S_0)
///   7. u = (I + A)^-1 rhs
///   8. o = exp(c) Q S_0 + M u, the decay having gone onto the queries rather than onto this
///   9. S <- exp(c_last) S + K~^T u, straight into the accumulator fragments
///
/// Sixteen warps, and both of the things that bound them are full: 128 registers a thread is the
/// whole register file, and shared memory is 87 KB of the 99 KB a block may have, so one CTA is
/// resident per SM. The half copy of the state and the two chunk-tall buffers overlap to get there
/// -- the state dies exactly where the right hand side is born.
namespace {

/// The chunk length. 64 rather than the fused FP32 path's 32: the tiles are what the tensor cores
/// want now, and a 64-row chunk is four of them down each side of the triangular system.
constexpr int kChunk = 64;

/// The WMMA tile. m16n16k16 is the only shape the half-in float-out API offers that is square,
/// which every product here wants since the same buffers are read as A, as B and transposed.
constexpr int kTile = 16;
constexpr int kBlocks = kChunk / kTile;

/// Sixteen warps. One CTA is resident per SM at this shared memory footprint, so the block's own
/// warps are the only thing the SM has to hide a shared memory latency, a barrier or an HMMA
/// dependency behind, and at eight warps -- two per scheduler -- it issued about a sixth of the
/// instruction slots it had. Sixteen is also the largest count the tiles still divide into evenly:
/// four state tiles and two chunk-tall tiles each, where twelve gives six to some warps and five to
/// others and was 5% slower than eight for it.
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;

/// Rows are padded by half a tile: a warp reading a column of a tile touches one row per bank
/// otherwise.
constexpr int kPad = 8;
constexpr int kScoreLd = kChunk + kPad;

constexpr int kMaxHeadDim = 128;

/// The state is (D / 16)^2 tiles over the warps, and the chunk-tall products are (64 / 16) * (D /
/// 16) of them. Both are handed out warp-strided, so these are the counts at the largest head
/// dimension.
constexpr int kMaxStateTiles =
    ((kMaxHeadDim / kTile) * (kMaxHeadDim / kTile) + kWarps - 1) / kWarps;
constexpr int kMaxOutTiles = (kBlocks * (kMaxHeadDim / kTile) + kWarps - 1) / kWarps;

/// The tiles of a chunk-square matrix that are at or below the diagonal, packed as (row, column).
/// Both A and the score matrix are zero above their diagonal, and every reader of either -- the
/// inversion's combines, the solve, the output -- is written to stay at or below it, so the six
/// tiles above it are zeroed once and never computed.
__device__ __constant__ unsigned char kLowerTiles[10] =
    {0x00, 0x10, 0x11, 0x20, 0x21, 0x22, 0x30, 0x31, 0x32, 0x33};

using FragA = wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half, wmma::row_major>;
using FragAT = wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half, wmma::col_major>;
using FragB = wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half, wmma::row_major>;
using FragBT = wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>;

/// Where a lane sits in a tile when the tile is read or written eight elements at a time: eight
/// consecutive columns of one row, two lanes to a row. Every epilogue below uses this, which is
/// what lets the scratch be read as float4 and the destination written 16 bytes at a time -- the
/// same work an element at a time is a third of the kernel's instructions.
/// Sixteen bytes from global straight into shared memory. A `bytes` of zero fills the destination
/// instead, which is how the rows past a short chunk's length are zeroed without a second pass.
__device__ inline void copyAsync16(half *dst, const half *src, int bytes) {
#if __CUDA_ARCH__ >= 800
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr), "l"(src), "r"(bytes));
#else
  // Before Ampere this is an ordinary load and store, which the barrier that follows the wait below
  // covers just as well -- it only stops being free.
  uint4 val = make_uint4(0, 0, 0, 0);
  if (bytes != 0) val = *reinterpret_cast<const uint4 *>(src);
  *reinterpret_cast<uint4 *>(dst) = val;
#endif
}

/// A barrier across the first four warps only. The inversion is theirs, and the point of it being
/// theirs is that the other four do not stop for it.
__device__ inline void barrierInverseWarps() {
  asm volatile("bar.sync 1, 128;\n" ::);
}

__device__ inline void copyAsyncWait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ inline int laneRow(int lane) {
  return lane >> 1;
}

__device__ inline int laneCol(int lane) {
  return (lane & 1) << 3;
}

__device__ inline void readScratch(const float *scratch, int lane, float *x) {
  const float4 *p = reinterpret_cast<const float4 *>(scratch + laneRow(lane) * kTile + laneCol(lane));
  float4 a = p[0];
  float4 b = p[1];
  x[0] = a.x;
  x[1] = a.y;
  x[2] = a.z;
  x[3] = a.w;
  x[4] = b.x;
  x[5] = b.y;
  x[6] = b.z;
  x[7] = b.w;
}

__device__ inline void writeHalf8(half *dst, const float *x) {
  half2 h[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(x[2 * i], x[2 * i + 1]);
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(h);
}

__device__ inline void readHalf8(const half *src, float *x) {
  uint4 raw = *reinterpret_cast<const uint4 *>(src);
  const half2 *h = reinterpret_cast<const half2 *>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __half22float2(h[i]);
    x[2 * i] = f.x;
    x[2 * i + 1] = f.y;
  }
}

/// Where a lane's eight accumulator elements sit in the tile, and whether they come in pairs that
/// share a row.
///
/// The WMMA API says a fragment's element order is unspecified, and every epilogue here needs to
/// know which (row, column) an element is. Rather than take the layout on trust -- it is an ABI
/// detail, not a contract -- the warp asks: it fills a scratch tile with each element's own index,
/// loads it as an accumulator, and reads back where each of its elements landed. Whatever the
/// answer is, it is right for the hardware the kernel is running on, and it costs one tile's worth
/// of shared memory traffic per warp per launch against the round trip it saves in every epilogue
/// of every chunk.
struct TileLayout {
  int row[8];
  int col[8];
  bool paired;
};

__device__ inline void probeTileLayout(float *scratch, int lane, TileLayout *out) {
  for (int e = lane; e < kTile * kTile; e += 32) scratch[e] = static_cast<float>(e);
  __syncwarp();

  FragC probe;
  wmma::load_matrix_sync(probe, scratch, kTile, wmma::mem_row_major);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int idx = static_cast<int>(probe.x[e]);
    out->row[e] = idx / kTile;
    out->col[e] = idx % kTile;
  }

  // Every layout in the field lays a lane's elements out as four pairs of adjacent columns, which
  // lets an epilogue write them 32 bits at a time. If some future one does not, the scalar path
  // below is still correct.
  bool paired = true;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    paired = paired && out->row[2 * j + 1] == out->row[2 * j] &&
             out->col[2 * j + 1] == out->col[2 * j] + 1;
  }
  out->paired = paired;
  __syncwarp();
}

/// Write a warp's eight accumulator values, already in whatever form the epilogue left them, into a
/// half matrix at the coordinates the probe found.
__device__ inline void scatterTileHalf(
    const TileLayout &layout,
    const float *x,
    half *dst,
    int ld) {
  if (layout.paired) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      half2 pair = __floats2half2_rn(x[2 * j], x[2 * j + 1]);
      *reinterpret_cast<half2 *>(dst + layout.row[2 * j] * ld + layout.col[2 * j]) = pair;
    }
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      dst[layout.row[e] * ld + layout.col[e]] = __float2half(x[e]);
    }
  }
}

/// Invert one 16 by 16 unit lower triangular block of `m` in place, by Gauss-Jordan with one lane
/// per row. Only the first 16 lanes take part.
///
/// Each lane holds its row of the working matrix, which starts as the row of (I + A) and ends as
/// the row of the inverse. Eliminating column `s` writes the multiplier into column `s` of every
/// row below it and folds that multiplier into the columns left of `s`, which are the ones the
/// inverse has already filled in. Column `s` of a row below is only read again by later
/// eliminations, by which time it holds its final value -- which is what makes one pass enough.
__device__ inline void invertUnitLowerTile(half *m, int ld, int lane) {
  constexpr unsigned kMask = 0xffffu;
  if (lane >= kTile) return;

  float row[kTile];
  readHalf8(m + lane * ld, row);
  readHalf8(m + lane * ld + 8, row + 8);
#pragma unroll
  for (int j = 0; j < kTile; ++j) {
    // The block arrives with a unit diagonal and zeros above it, but taking that on trust here
    // would make this depend on a mask two phases away.
    if (lane == j) {
      row[j] = 1.0f;
    } else if (lane < j) {
      row[j] = 0.0f;
    }
  }

#pragma unroll
  for (int s = 0; s < kTile - 1; ++s) {
    float scale = -row[s];
#pragma unroll
    for (int i = 0; i < s; ++i) {
      // The shuffle is outside the branch: every lane of the mask has to reach it. Only the columns
      // left of the one being eliminated hold anything yet, which is what keeps this at half the
      // shuffles a square loop would do.
      float above = __shfl_sync(kMask, row[i], s);
      if (lane > s) row[i] += scale * above;
    }
    if (lane > s) row[s] = scale;
  }

  writeHalf8(m + lane * ld, row);
  writeHalf8(m + lane * ld + 8, row + 8);
}

/// C = A B over `numK` tiles of the contraction, with both operands row major in shared memory.
/// The result is left in the accumulator rather than stored, since every caller has an epilogue.
__device__ inline void gemmTiles(
    FragC &acc,
    const half *a,
    int lda,
    const half *b,
    int ldb,
    int numK) {
  for (int kk = 0; kk < numK; ++kk) {
    FragA fa;
    FragB fb;
    wmma::load_matrix_sync(fa, a + kk * kTile, lda);
    wmma::load_matrix_sync(fb, b + kk * kTile * ldb, ldb);
    wmma::mma_sync(acc, fa, fb, acc);
  }
}

/// Write an accumulator tile into a half matrix, optionally negated. The scratch is this warp's
/// own, so the only ordering this needs is against itself.
__device__ inline void storeTileHalf(
    const TileLayout &layout,
    const FragC &acc,
    half *dst,
    int ld,
    bool negate) {
  float x[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) x[e] = negate ? -acc.x[e] : acc.x[e];
  scatterTileHalf(layout, x, dst, ld);
}

__global__ __launch_bounds__(kThreads) void gatedDeltaNetWmmaKernel(
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
    int d,
    int headRatio) {
  const int head = blockIdx.x;
  const int seq = blockIdx.y;
  const int kHead = head / headRatio;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  const int begin = cuSeqlens[seq];
  const int end = cuSeqlens[seq + 1];
  if (begin >= end) return;

  const int ld = d + kPad;
  const int nb = d / kTile;
  const int numStateTiles = nb * nb;
  const int numOutTiles = kBlocks * nb;

  // Tiles are handed out warp-strided rather than in consecutive runs, which spreads a warp's tiles
  // over the row blocks. That is what keeps the two triangular phases balanced: a tile in row block
  // r costs r + 1 steps of the contraction, so a warp holding one row block would hold either the
  // cheapest tiles or the dearest.
  //
  // Two rearrangements of this were tried and both lost. Handing out consecutive tiles gives a warp
  // one row block, which shares the operand that row block indexes and reads 25% fewer fragments --
  // and is 9% slower, because of the imbalance above. Keeping the strided tiles but hoisting the
  // operand the fixed column block indexes out of the tile loop reads 25% fewer fragments too, and
  // is 12% slower: it puts a fragment across the loop and takes the fragment loads of the tiles out
  // of the shadow of each other's HMMAs.

  extern __shared__ char smemRaw[];
  float *sCum = reinterpret_cast<float *>(smemRaw);
  float *sCexp = sCum + kChunk;
  float *sClast = sCexp + kChunk;
  float *sBeta = sClast + kChunk;
  half *sK = reinterpret_cast<half *>(sBeta + kChunk);
  half *sQ = sK + kChunk * ld;
  half *sScore = sQ + kChunk * ld;
  half *sInv = sScore + kChunk * kScoreLd;

  // The state's half copy is read by phase 5 and dead the moment phase 6 starts writing the right
  // hand side, so the two share their storage. The output is staged over Q, which phase 5 is also
  // the last to read.
  half *sState = sInv + kChunk * kScoreLd;
  half *sRhs = sState;
  half *sU = sRhs + kChunk * ld;
  // The probe wants a tile of float scratch per warp and is done with it before the first chunk, so
  // it borrows the two chunk-square buffers rather than keeping a tile of its own for the whole
  // launch: they are written per chunk, and the first of those writes is after this.
  TileLayout layout;
  probeTileLayout(
      reinterpret_cast<float *>(sScore) + warp * kTile * kTile,
      lane,
      &layout);

  const int64_t stateOffset =
      (static_cast<int64_t>(stateSlots[seq]) * numVHead + head) * d * d;

  // The state, as the accumulator it stays in for the whole sequence. Tile t of it is warp
  // t % kWarps's, which is the same rule every phase below hands out tiles by.
  FragC accState[kMaxStateTiles];
#pragma unroll
  for (int i = 0; i < kMaxStateTiles; ++i) {
    int t = warp + i * kWarps;
    if (t < numStateTiles) {
      wmma::load_matrix_sync(
          accState[i],
          state + stateOffset + static_cast<int64_t>(t / nb) * kTile * d + (t % nb) * kTile,
          d,
          wmma::mem_row_major);
    }
  }

  // The probe above wrote over the two buffers zeroed below, a tile to a warp, so nobody may start
  // zeroing until every warp has read its answer back.
  __syncthreads();

  // The six tiles above the diagonal of A and of the score matrix, which stay zero for the whole
  // sequence: the inversion reads them as the zero blocks of a block triangular matrix, and nothing
  // writes them.
  for (int idx = tid; idx < kChunk * kBlocks * kTile / 8; idx += kThreads) {
    int i = idx / (kBlocks * kTile / 8);
    int j = (idx - i * (kBlocks * kTile / 8)) * 8;
    if (j > i) {
      const uint4 zero = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(sScore + i * kScoreLd + j) = zero;
      *reinterpret_cast<uint4 *>(sInv + i * kScoreLd + j) = zero;
    }
  }

  for (int t0 = begin; t0 < end; t0 += kChunk) {
    const int len = min(kChunk, end - t0);

    // -- 0. the cumulative log decay, and the three arrays the epilogues below read by row. One warp
    // holds both halves of the chunk, a lane to each of two rows, so the scan, its carry and the
    // chunk's last cumulative decay are all shuffles -- and the phase needs no barrier of its own,
    // since the one after the state conversion below already stands between it and its first reader.
    __syncthreads();
    if (warp == 0) {
      const int hi = lane + 32;
      float x0 = (lane < len) ? g[static_cast<int64_t>(t0 + lane) * numVHead + head] : 0.0f;
      float x1 = (hi < len) ? g[static_cast<int64_t>(t0 + hi) * numVHead + head] : 0.0f;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        float y0 = __shfl_up_sync(0xffffffff, x0, offset);
        float y1 = __shfl_up_sync(0xffffffff, x1, offset);
        if (lane >= offset) {
          x0 += y0;
          x1 += y1;
        }
      }
      x1 += __shfl_sync(0xffffffff, x0, 31);

      float last = (len > 32) ? __shfl_sync(0xffffffff, x1, len - 33)
                              : __shfl_sync(0xffffffff, x0, len - 1);
      sCum[lane] = x0;
      sCum[hi] = x1;
      sCexp[lane] = (lane < len) ? expf(x0) : 0.0f;
      sCexp[hi] = (hi < len) ? expf(x1) : 0.0f;
      sClast[lane] = (lane < len) ? expf(last - x0) : 0.0f;
      sClast[hi] = (hi < len) ? expf(last - x1) : 0.0f;
      sBeta[lane] = (lane < len) ? beta[static_cast<int64_t>(t0 + lane) * numVHead + head] : 0.0f;
      sBeta[hi] = (hi < len) ? beta[static_cast<int64_t>(t0 + hi) * numVHead + head] : 0.0f;
    }

    // -- 1. K and Q. Rows past the chunk's length are zeroed rather than masked later: a zero row
    // of K takes A, the score matrix, the right hand side and the state update out in one go.
    const int vecPerRow = d / 8;
    for (int idx = tid; idx < kChunk * vecPerRow; idx += kThreads) {
      int i = idx / vecPerRow;
      int m = (idx - i * vecPerRow) * 8;
      int64_t off = (static_cast<int64_t>(t0 + i) * numKHead + kHead) * d + m;
      int bytes = (i < len) ? 16 : 0;
      copyAsync16(sK + i * ld + m, k + off, bytes);
      copyAsync16(sQ + i * ld + m, q + off, bytes);
    }

    // -- 2. the state, out of the fragments and into shared memory as half. This is the only way
    // the phase that reads it can address it by coordinate, and it is once per chunk against the
    // two full passes it feeds.
#pragma unroll
    for (int i = 0; i < kMaxStateTiles; ++i) {
      int t = warp + i * kWarps;
      if (t >= numStateTiles) break;
      storeTileHalf(layout, accState[i], sState + (t / nb) * kTile * ld + (t % nb) * kTile, ld, false);
    }

    // Nothing before this wanted the keys or the queries, so their arrival waits behind the state
    // conversion rather than in front of it.
    copyAsyncWait();
    __syncthreads();

    // -- 3. A and the score matrix. Both are a chunk by chunk product against K^T, so they are one
    // pass with two accumulators over the same B fragment.
    for (int t = warp; t < 10; t += kWarps) {
      int rb = kLowerTiles[t] >> 4;
      int cb = kLowerTiles[t] & 0xf;

      FragC accA;
      FragC accM;
      wmma::fill_fragment(accA, 0.0f);
      wmma::fill_fragment(accM, 0.0f);
      for (int kk = 0; kk < nb; ++kk) {
        FragA fk;
        FragA fq;
        FragBT fb;
        wmma::load_matrix_sync(fk, sK + rb * kTile * ld + kk * kTile, ld);
        wmma::load_matrix_sync(fq, sQ + rb * kTile * ld + kk * kTile, ld);
        wmma::load_matrix_sync(fb, sK + cb * kTile * ld + kk * kTile, ld);
        wmma::mma_sync(accA, fk, fb, accA);
        wmma::mma_sync(accM, fq, fb, accM);
      }

      // Both matrices carry the same exp(c_i - c_j), so they share the epilogue as well as the K^T
      // fragment: doing them apart is the same decay computed twice.
      float a[8];
      float m[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int i = rb * kTile + layout.row[e];
        int j = cb * kTile + layout.col[e];

        // Taken as a difference of logs, which is the only form that is safe: either exponential on
        // its own can be far outside half's range where the ratio is not.
        float decay = (i < len && j <= i) ? expf(sCum[i] - sCum[j]) : 0.0f;
        m[e] = decay * accM.x[e];
        // The identity part of (I + A), on every row, so the rows a short chunk does not use invert
        // to themselves.
        a[e] = (i == j) ? 1.0f : sBeta[i] * decay * accA.x[e];
      }

      scatterTileHalf(layout, m, sScore + rb * kTile * kScoreLd + cb * kTile, kScoreLd);
      scatterTileHalf(layout, a, sInv + rb * kTile * kScoreLd + cb * kTile, kScoreLd);
    }
    __syncthreads();

    // The output's first term is exp(c_i) q_i S_0, and scaling the queries here is what keeps it
    // out of the accumulator's way: phase 8 can then let the score matrix's term accumulate
    // straight on top of it instead of taking the fragment out to shared memory to scale it and
    // putting it back. The scale is at most one, so what it risks is a query row flushing to zero
    // where exp(c_i) is tiny -- which is a term negligible against the same row's M u by exactly
    // the factor that flushed it. The score matrix is already built, and read the queries unscaled.
    for (int idx = tid; idx < kChunk * vecPerRow; idx += kThreads) {
      int i = idx / vecPerRow;
      int m = (idx - i * vecPerRow) * 8;
      float x[8];
      readHalf8(sQ + i * ld + m, x);
      float decay = sCexp[i];
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] *= decay;
      writeHalf8(sQ + i * ld + m, x);
    }

    // The queries are wanted scaled from here on, and the four warps that skip the inversion below
    // go straight to reading them, so this is where everyone has to agree on them.
    __syncthreads();

    // -- 4. the inverse of (I + A), in place, on the first four warps alone -- and the other four do
    // not wait for it. Nothing in phase 5 reads the inverse, and phase 7, which does, is four block
    // barriers further on; what the split buys is that the shuffles and the small dependent products
    // here run against the other half of the block's HMMAs rather than in front of them.
    //
    // The diagonal blocks go first, by elimination; the two combines are what puts the rest of it on
    // the tensor cores.
    if (warp < kBlocks) {
      invertUnitLowerTile(sInv + warp * kTile * kScoreLd + warp * kTile, kScoreLd, lane);
      barrierInverseWarps();

      // 16 to 32, on the two diagonal 32 by 32 blocks: the off-diagonal tile of the inverse is
      // -T11 A10 T00, and the tile it overwrites is the A10 it is built from, so the warp that owns
      // it does both products and nobody else touches it.
      if (warp < 2) {
        int r = 2 * warp + 1;
        int c = 2 * warp;
        half *dst = sInv + r * kTile * kScoreLd + c * kTile;

        FragC acc;
        wmma::fill_fragment(acc, 0.0f);
        gemmTiles(acc, dst, kScoreLd, sInv + c * kTile * kScoreLd + c * kTile, kScoreLd, 1);
        storeTileHalf(layout, acc, dst, kScoreLd, false);

        wmma::fill_fragment(acc, 0.0f);
        gemmTiles(acc, sInv + r * kTile * kScoreLd + r * kTile, kScoreLd, dst, kScoreLd, 1);
        storeTileHalf(layout, acc, dst, kScoreLd, true);
      }
      barrierInverseWarps();

      // 32 to 64. The lower left 32 by 32 is four tiles over the four warps, and this time the
      // product reads tiles another warp is about to overwrite, so the store waits for all of them.
      const int invR = 2 + warp / 2;
      const int invC = warp % 2;
      FragC accInv;
      wmma::fill_fragment(accInv, 0.0f);
      gemmTiles(accInv, sInv + invR * kTile * kScoreLd, kScoreLd, sInv + invC * kTile, kScoreLd, 2);
      barrierInverseWarps();

      storeTileHalf(layout, accInv, sInv + invR * kTile * kScoreLd + invC * kTile, kScoreLd, false);
      barrierInverseWarps();

      wmma::fill_fragment(accInv, 0.0f);
      gemmTiles(
          accInv,
          sInv + invR * kTile * kScoreLd + 2 * kTile,
          kScoreLd,
          sInv + 2 * kTile * kScoreLd + invC * kTile,
          kScoreLd,
          2);
      barrierInverseWarps();

      storeTileHalf(layout, accInv, sInv + invR * kTile * kScoreLd + invC * kTile, kScoreLd, true);
    }

    // -- 5. K S_0 and Q S_0. Half of the chunk's arithmetic is here, and both products share the
    // state's B fragment, so they are one pass over it.
    FragC accKs[kMaxOutTiles];
    FragC accQs[kMaxOutTiles];
#pragma unroll
    for (int i = 0; i < kMaxOutTiles; ++i) {
      wmma::fill_fragment(accKs[i], 0.0f);
      wmma::fill_fragment(accQs[i], 0.0f);
    }
#pragma unroll
    for (int i = 0; i < kMaxOutTiles; ++i) {
      int t = warp + i * kWarps;
      if (t >= numOutTiles) break;

      int rb = t / nb;
      int cb = t % nb;
      for (int kk = 0; kk < nb; ++kk) {
        FragA fk;
        FragA fq;
        FragB fs;
        wmma::load_matrix_sync(fk, sK + rb * kTile * ld + kk * kTile, ld);
        wmma::load_matrix_sync(fq, sQ + rb * kTile * ld + kk * kTile, ld);
        wmma::load_matrix_sync(fs, sState + kk * kTile * ld + cb * kTile, ld);
        wmma::mma_sync(accKs[i], fk, fs, accKs[i]);
        wmma::mma_sync(accQs[i], fq, fs, accQs[i]);
      }
    }
    // The right hand side lands on top of the state's half copy, so every warp has to be done
    // reading it first.
    __syncthreads();

    // v goes through the right hand side's own buffer on the way in. Read where phase 6 wants it,
    // a lane's eight columns of one row, it would be sixteen 32-byte sectors a warp; read here it
    // is the same coalesced pass the keys and queries get, and the epilogue below picks it up from
    // shared memory and overwrites it in place.
    for (int idx = tid; idx < kChunk * vecPerRow; idx += kThreads) {
      int i = idx / vecPerRow;
      int m = (idx - i * vecPerRow) * 8;
      uint4 vv = make_uint4(0, 0, 0, 0);
      if (i < len) {
        vv = *reinterpret_cast<const uint4 *>(
            v + (static_cast<int64_t>(t0 + i) * numVHead + head) * d + m);
      }
      *reinterpret_cast<uint4 *>(sRhs + i * ld + m) = vv;
    }
    __syncthreads();

    // -- 6. the right hand side, beta (v - exp(c) K S_0).
#pragma unroll
    for (int i = 0; i < kMaxOutTiles; ++i) {
      int t = warp + i * kWarps;
      if (t >= numOutTiles) break;

      int rb = t / nb;
      int cb = t % nb;
      float rhs[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = rb * kTile + layout.row[e];
        int col = cb * kTile + layout.col[e];
        float vv = __half2float(sRhs[row * ld + col]);
        rhs[e] = (row < len) ? sBeta[row] * (vv - sCexp[row] * accKs[i].x[e]) : 0.0f;
      }

      // In place over v, which this warp's own lanes are the only readers of.
      scatterTileHalf(layout, rhs, sRhs + rb * kTile * ld + cb * kTile, ld);
    }
    __syncthreads();

    // -- 7. u. The inverse is lower triangular, so a tile row only reads the tiles at or left of
    // its diagonal.
    for (int t = warp; t < numOutTiles; t += kWarps) {
      int rb = t / nb;
      int cb = t % nb;

      FragC acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int kk = 0; kk <= rb; ++kk) {
        FragA fa;
        FragB fb;
        wmma::load_matrix_sync(fa, sInv + rb * kTile * kScoreLd + kk * kTile, kScoreLd);
        wmma::load_matrix_sync(fb, sRhs + kk * kTile * ld + cb * kTile, ld);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      storeTileHalf(layout, acc, sU + rb * kTile * ld + cb * kTile, ld, false);
    }
    __syncthreads();

    // -- 8. the output. exp(c) Q S_0 has been in this warp's accumulator since phase 5 -- the decay
    // went onto the queries before that product rather than onto its result -- so the score
    // matrix's part accumulates straight on top of it and the fragment is only read once.
#pragma unroll
    for (int i = 0; i < kMaxOutTiles; ++i) {
      int t = warp + i * kWarps;
      if (t >= numOutTiles) break;

      int rb = t / nb;
      int cb = t % nb;

      for (int kk = 0; kk <= rb; ++kk) {
        FragA fa;
        FragB fb;
        wmma::load_matrix_sync(fa, sScore + rb * kTile * kScoreLd + kk * kTile, kScoreLd);
        wmma::load_matrix_sync(fb, sU + kk * kTile * ld + cb * kTile, ld);
        wmma::mma_sync(accQs[i], fa, fb, accQs[i]);
      }

      // Straight out to global from the fragment, four bytes at a time. A warp's lanes cover a row
      // of the tile between them, so the four stores are four coalesced halves of it rather than
      // one, which is the price of not keeping a scratch tile for the whole launch.
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int row = rb * kTile + layout.row[2 * j];
        int col = cb * kTile + layout.col[2 * j];
        if (row < len) {
          *reinterpret_cast<half2 *>(
              out + (static_cast<int64_t>(t0 + row) * numVHead + head) * d + col) =
              __floats2half2_rn(accQs[i].x[2 * j], accQs[i].x[2 * j + 1]);
        }
      }
    }

    // -- 9. the state. K carries the decay of the chunk's tail into the product, which is the last
    // thing either of them is needed for.
    for (int idx = tid; idx < kChunk * vecPerRow; idx += kThreads) {
      int i = idx / vecPerRow;
      int m = (idx - i * vecPerRow) * 8;
      float x[8];
      readHalf8(sK + i * ld + m, x);
      float decay = sClast[i];
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] *= decay;
      writeHalf8(sK + i * ld + m, x);
    }
    __syncthreads();

    const float gamma = sCexp[len - 1];
#pragma unroll
    for (int i = 0; i < kMaxStateTiles; ++i) {
      if (warp + i * kWarps >= numStateTiles) break;
#pragma unroll
      for (int e = 0; e < accState[i].num_elements; ++e) accState[i].x[e] *= gamma;
    }

#pragma unroll
    for (int i = 0; i < kMaxStateTiles; ++i) {
      int t = warp + i * kWarps;
      if (t >= numStateTiles) break;

      int rb = t / nb;
      int cb = t % nb;
      for (int kk = 0; kk < kBlocks; ++kk) {
        FragAT fa;
        FragB fb;
        wmma::load_matrix_sync(fa, sK + kk * kTile * ld + rb * kTile, ld);
        wmma::load_matrix_sync(fb, sU + kk * kTile * ld + cb * kTile, ld);
        wmma::mma_sync(accState[i], fa, fb, accState[i]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kMaxStateTiles; ++i) {
    int t = warp + i * kWarps;
    if (t < numStateTiles) {
      wmma::store_matrix_sync(
          state + stateOffset + static_cast<int64_t>(t / nb) * kTile * d + (t % nb) * kTile,
          accState[i],
          d,
          wmma::mem_row_major);
    }
  }
}

size_t sharedBytes(int d) {
  int ld = d + kPad;
  size_t floats = 4 * kChunk;
  size_t halves = static_cast<size_t>(kChunk) * ld * 2 +   // sK, sQ
                  static_cast<size_t>(kChunk) * kScoreLd * 2 +  // sScore, sInv
                  static_cast<size_t>(2 * kChunk) * ld;    // sState, over which sRhs and sU lie
  return floats * sizeof(float) + halves * sizeof(half);
}

}  // namespace

bool fits(int headDim, int *smemOut) {
  if (headDim % kTile != 0 || headDim > kMaxHeadDim) return false;

  size_t smem = sharedBytes(headDim);
  int maxSmem = 0;
  LL_CHECK_CUDA_STATUS(
      cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
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
      << "the tensor core gatedDeltaNetPrefill needs a head dimension that is a multiple of "
      << kTile << " and at most " << kMaxHeadDim << ", got " << headDim;

  LL_CHECK_CUDA_STATUS(cudaFuncSetAttribute(
      gatedDeltaNetWmmaKernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem));

  gatedDeltaNetWmmaKernel<<<dim3(numVHead, numSeq), kThreads, smem>>>(
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

}  // namespace gdnwmma
}  // namespace cuda
}  // namespace op
}  // namespace fl
