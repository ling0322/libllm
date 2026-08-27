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

#include <algorithm>

#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {
namespace gdnmma {

/// The gated DeltaNet prefill with nothing but operands in shared memory.
///
/// The WMMA path next door already puts every product on the tensor cores, and it is four to five
/// times faster than the FP32 ones for it. What it cannot do is keep an accumulator: the API says a
/// fragment's element order is unspecified, so anything a product computes has to go out to shared
/// memory before another product can read it as an operand. The state goes out once a chunk, the
/// right hand side and u once each, and every epilogue reads a tile back by coordinate.
///
/// `mma.sync` says exactly where each element is, and one consequence is worth the whole rewrite:
/// two adjacent m16n8 accumulators, cast to half, *are* the A operand of the 16 by 16 tile they
/// cover. So a product's result can feed the next product straight out of the registers it landed
/// in. Held that way, the state, the right hand side, u and the output never touch shared memory,
/// which leaves it holding only what is genuinely an operand: K, Q, the inverse and the score
/// matrix.
///
/// The rearrangement that makes that work is transposing the state, which is also how FlashInfer's
/// kernel holds it. With M = S^T of shape (value dim, key dim):
///
///   K S_0    becomes   M K^T,      Q S_0   becomes  M Q^T
///   u        becomes   u^T = rhs^T (I + A)^-T
///   o        becomes   o^T = exp(c) M Q^T + u^T M_score^T
///   S update becomes   M <- exp(c_last) M + (u^T diag(exp(c_last - c))) K
///
/// and every one of those has the thing computed before it as its A operand. It also puts the
/// token index on the *column* of every accumulator, so the two decays that the WMMA path applies
/// as separate in-place passes over shared memory -- exp(c) on the queries and exp(c_last - c) on
/// the keys -- become a multiply on a register, with no pass and no barrier.
///
/// A warp owns 16 rows of the value dimension and every column of them: the state's 16 by D, the
/// chunk-tall 16 by 64 of K S_0, of u and of the output. Nothing it needs is ever partitioned
/// along an axis it contracts over, so there is no cross-warp reduction anywhere.
///
/// What is still in shared memory, and why it has to be: K and Q are read as B operands by
/// everything, the inverse and the score matrix are shared across warps, and v arrives transposed
/// against the way the epilogue wants it, so it is staged coalesced and read back a lane at a time.
/// That staging buffer is also where the output is assembled, since o^T comes out with the token on
/// the column and has to leave with the token on the row.
///
/// The keys get two buffers rather than one, so that the next chunk's copies can be issued from
/// inside this one -- the queries as soon as the state products have read them, the values once the
/// output has been written out of the buffer they share. What is left waiting at the top of a chunk
/// is what could not be started earlier, which is very little.
///
/// A sequence short enough to be a decode step takes a different path through the same kernel. The
/// chunk machinery costs the same whether the chunk holds one token or sixty-four -- twenty tile
/// products for K K^T and Q K^T, a 64 by 64 inversion, and the whole width of every accumulator --
/// so a batch of one-token sequences pays sixty-four times over for what a rank-one update does.
/// Splitting it off is what vLLM does too, but it does it by reordering the batch and launching a
/// second kernel, which it can afford because its scheduler already knows every length on the host.
/// Here the lengths are only on the device, and a second launch over the same grid is not free: a
/// CTA that reads its length and returns still has to be scheduled and still holds this kernel's
/// ~87KB of shared memory while it does, which measured 21us for a grid of 4096 on an RTX 5060 Ti.
/// So the branch is per CTA, inside the one launch, and neither path pays for the other.
///
/// The recurrent path costs nothing extra to hold: the state is already in the registers it needs
/// it in, and the layout is already the one that makes the step warp-local. A warp owns sixteen
/// rows of the value dimension and every key-dimension column of them, and the step
///
///   h = exp(g) S^T k,   u = beta (v - h),   S <- exp(g) S + k u^T,   o = S^T q
///
/// contracts over the key dimension only -- which is the accumulator's *column*, split four ways
/// across lanes and no further. So h and o each end in two butterfly shuffles over lanes ^1 and ^2,
/// u never leaves the four lanes that need it, and the update is a multiply-add on a register.
/// There is no shared memory in the step and no barrier in the loop; the tokens are staged once,
/// into the buffers the chunk path would have used, and the state is read and written exactly as
/// often as the chunk path reads and writes it, which is once each.
namespace {

constexpr int kChunk = 64;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

/// The mma shape. m16n8k16 is the half-in float-out instruction on this architecture; the 16 by 16
/// A operand is two of its accumulators side by side, which is the identity the kernel is built on.
constexpr int kTileM = 16;
constexpr int kTileN = 8;
constexpr int kTileK = 16;

constexpr int kNTiles = kChunk / kTileN;   // token tiles across an accumulator
constexpr int kKBlocks = kChunk / kTileK;  // 16-token blocks of a contraction over the chunk

constexpr int kMaxHeadDim = 128;
constexpr int kMaxNTilesD = kMaxHeadDim / kTileN;

constexpr int kPad = 8;
constexpr int kScoreLd = kChunk + kPad;

/// One m16n8 accumulator: rows m, columns n, four floats a lane.
struct Acc {
  float x[4];
};

/// Where a lane's four accumulator elements sit, straight out of the PTX ABI (and checked against a
/// reference GEMM on this device). Elements 0 and 1 are adjacent columns of row `accRow`, elements
/// 2 and 3 the same columns eight rows down.
__device__ inline int accRow(int lane) {
  return lane >> 2;
}

__device__ inline int accCol(int lane) {
  return (lane & 3) << 1;
}

__device__ inline void mma16816(Acc &d, const unsigned *a, const unsigned *b) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

/// A 16 by 16 tile of a row-major shared matrix as an A operand: lane l takes row l % 16 of the
/// tile and the half of the contraction l / 16 sits in.
__device__ inline void loadA(unsigned *r, const half *src, int ld, int lane) {
#if __CUDA_ARCH__ >= 800
  const half *p = src + (lane & 15) * ld + (lane >> 4) * 8;
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

/// Two B operands, sixteen columns of them, out of a matrix stored with those columns as its rows
/// -- which is what every B operand here has, since the contraction of each product runs along the
/// row of the buffer it reads.
__device__ inline void loadB(unsigned *r, const half *src, int ld, int lane) {
  loadA(r, src, ld, lane);
}

/// Two B operands out of a matrix stored the other way round, transposed on the way in.
__device__ inline void loadBTrans(unsigned *r, const half *src, int ld, int lane) {
#if __CUDA_ARCH__ >= 800
  const half *p = src + (lane & 15) * ld + (lane >> 4) * 8;
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

/// The two B operands of a 16-column block that `loadB` fetched. `ldmatrix` hands its four
/// registers back as (n 0-7, k 0-7), (n 8-15, k 0-7), (n 0-7, k 8-15), (n 8-15, k 8-15) -- the
/// halves of the address pattern index k -- so an operand, which needs all sixteen of the
/// contraction for eight columns, is registers i and i + 2.
__device__ inline void mmaB(Acc &lo, Acc &hi, const unsigned *a, const unsigned *b4) {
  unsigned b[2];
  b[0] = b4[0];
  b[1] = b4[2];
  mma16816(lo, a, b);
  b[0] = b4[1];
  b[1] = b4[3];
  mma16816(hi, a, b);
}

/// The same for a block `loadBTrans` fetched, where it is the other way round: the transpose makes
/// the halves of the address pattern index n, so an operand is registers 2i and 2i + 1. Getting
/// these two the wrong way round is silent -- the shapes are identical and only the answer changes
/// -- so they are the only place either pairing is written down.
__device__ inline void mmaBTrans(Acc &lo, Acc &hi, const unsigned *a, const unsigned *b4) {
  mma16816(lo, a, b4);
  mma16816(hi, a, b4 + 2);
}

/// The identity the kernel is built on: the pair of accumulators covering columns [0, 8) and
/// [8, 16) of the same sixteen rows, cast to half, is the A operand of that 16 by 16 tile.
__device__ inline void accToOperand(unsigned *a, const Acc &lo, const Acc &hi) {
  half2 *h = reinterpret_cast<half2 *>(a);
  h[0] = __floats2half2_rn(lo.x[0], lo.x[1]);
  h[1] = __floats2half2_rn(lo.x[2], lo.x[3]);
  h[2] = __floats2half2_rn(hi.x[0], hi.x[1]);
  h[3] = __floats2half2_rn(hi.x[2], hi.x[3]);
}

__device__ inline void copyAsync16(half *dst, const half *src, int bytes) {
#if __CUDA_ARCH__ >= 800
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr), "l"(src), "r"(bytes));
#else
  uint4 val = make_uint4(0, 0, 0, 0);
  if (bytes != 0) val = *reinterpret_cast<const uint4 *>(src);
  *reinterpret_cast<uint4 *>(dst) = val;
#endif
}

__device__ inline void copyAsyncCommit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ inline void copyAsyncWait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

/// A barrier across the first four warps, which own the inversion the other four walk past.
__device__ inline void barrierInverseWarps() {
  asm volatile("bar.sync 1, 128;\n" ::);
}

/// Invert one 16 by 16 unit lower triangular block in place by Gauss-Jordan, a lane to a row. See
/// the WMMA path for why one pass over the columns is enough.
__device__ inline void invertUnitLowerTile(half *m, int ld, int lane) {
  constexpr unsigned kMask = 0xffffu;
  if (lane >= kTileM) return;

  float row[kTileM];
#pragma unroll
  for (int j = 0; j < kTileM; ++j) {
    if (lane == j) {
      row[j] = 1.0f;
    } else if (lane < j) {
      row[j] = 0.0f;
    } else {
      row[j] = __half2float(m[lane * ld + j]);
    }
  }

#pragma unroll
  for (int s = 0; s < kTileM - 1; ++s) {
    float scale = -row[s];
#pragma unroll
    for (int i = 0; i < s; ++i) {
      float above = __shfl_sync(kMask, row[i], s);
      if (lane > s) row[i] += scale * above;
    }
    if (lane > s) row[s] = scale;
  }

#pragma unroll
  for (int j = 0; j < kTileM; ++j) m[lane * ld + j] = __float2half(row[j]);
}

/// C = A B over one 16 by 16 tile of a lower triangular matrix's own storage, used only by the
/// inversion's combines.
__device__ inline void tileGemm(
    Acc *acc,
    const half *a,
    int lda,
    const half *b,
    int ldb,
    int numK,
    int lane) {
  acc[0] = {0.0f, 0.0f, 0.0f, 0.0f};
  acc[1] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int kk = 0; kk < numK; ++kk) {
    unsigned fa[4];
    unsigned fb[4];
    loadA(fa, a + kk * kTileK, lda, lane);
    loadBTrans(fb, b + kk * kTileK * ldb, ldb, lane);
    mmaBTrans(acc[0], acc[1], fa, fb);
  }
}

/// Write a 16 by 16 result, held as two accumulators, into a half matrix.
__device__ inline void storeTile(const Acc *acc, half *dst, int ld, int lane, bool negate) {
  int r = accRow(lane);
  int c = accCol(lane);
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    float a0 = acc[t].x[0];
    float a1 = acc[t].x[1];
    float a2 = acc[t].x[2];
    float a3 = acc[t].x[3];
    if (negate) {
      a0 = -a0;
      a1 = -a1;
      a2 = -a2;
      a3 = -a3;
    }
    *reinterpret_cast<half2 *>(dst + r * ld + c + t * 8) = __floats2half2_rn(a0, a1);
    *reinterpret_cast<half2 *>(dst + (r + 8) * ld + c + t * 8) = __floats2half2_rn(a2, a3);
  }
}

/// One short sequence, stepped rather than chunked, over the state the caller already has in
/// registers. `stage` is `kMaxRecurrentLen * kD` floats four times over -- k, q, v and the output --
/// which is why the length is capped; it is carved out of the buffers the chunk path stages K, Q
/// and v in, and those are far larger than this needs.
///
/// The whole step lives inside a warp. The contraction of both products runs along the key
/// dimension, which is the accumulator's column and is split across four lanes and no further, so
/// each reduction is two butterfly shuffles; u is a function of the value-dimension row, which one
/// warp owns outright. Nothing here is shared between warps, so the only barriers are the two
/// around the staging buffer.
template <int kD>
__device__ inline void recurrentSequence(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    const float *__restrict__ g,
    const float *__restrict__ beta,
    half *__restrict__ out,
    Acc *accState,
    float *sExpG,
    float *sBeta,
    float *stage,
    int begin,
    int len,
    int head,
    int kHead,
    int numKHead,
    int numVHead,
    int tid,
    int lane,
    int row0,
    bool ownsRows) {
  constexpr int d = kD;
  constexpr int nTilesD = d / kTileN;

  // The staging buffer is the chunk path's K, Q and v buffers, which it is not using here.
  static_assert(
      4 * kMaxRecurrentLen * kD * sizeof(float) <= 4 * kChunk * (kD + kPad) * sizeof(half),
      "the recurrent path's staging does not fit the buffers it borrows");

  float *tK = stage;
  float *tQ = tK + kMaxRecurrentLen * d;
  float *tV = tQ + kMaxRecurrentLen * d;
  float *tO = tV + kMaxRecurrentLen * d;

  // The tokens, once. The loop below reads each of them from shared memory a lane at a time and in
  // an order that has nothing to do with the one they arrive in, so they are staged coalesced here
  // rather than read where they are used.
  for (int idx = tid; idx < len * d; idx += kThreads) {
    int i = idx / d;
    int c = idx - i * d;
    int64_t kqOff = (static_cast<int64_t>(begin + i) * numKHead + kHead) * d + c;
    tK[idx] = __half2float(k[kqOff]);
    tQ[idx] = __half2float(q[kqOff]);
    tV[idx] = __half2float(v[(static_cast<int64_t>(begin + i) * numVHead + head) * d + c]);
  }
  for (int i = tid; i < len; i += kThreads) {
    int64_t off = static_cast<int64_t>(begin + i) * numVHead + head;
    sExpG[i] = expf(g[off]);
    sBeta[i] = beta[off];
  }
  __syncthreads();

  if (ownsRows) {
    // This lane's two rows of the value dimension, and the two key-dimension columns it holds of
    // every eight. Both are fixed for the whole sequence.
    const int r = accRow(lane);
    const int c = accCol(lane);

    for (int i = 0; i < len; ++i) {
      const float a = sExpG[i];
      const float b = sBeta[i];
      const float *tk = tK + i * d;
      const float *tq = tQ + i * d;

      // h = S^T k, still undecayed: the decay is a scalar and rides along to where u needs it,
      // which saves a pass over the state that applying it here would cost.
      float h0 = 0.0f;
      float h1 = 0.0f;
#pragma unroll
      for (int t = 0; t < nTilesD; ++t) {
        float k0 = tk[t * kTileN + c];
        float k1 = tk[t * kTileN + c + 1];
        h0 += accState[t].x[0] * k0 + accState[t].x[1] * k1;
        h1 += accState[t].x[2] * k0 + accState[t].x[3] * k1;
      }
      h0 += __shfl_xor_sync(0xffffffff, h0, 1);
      h1 += __shfl_xor_sync(0xffffffff, h1, 1);
      h0 += __shfl_xor_sync(0xffffffff, h0, 2);
      h1 += __shfl_xor_sync(0xffffffff, h1, 2);

      const float u0 = b * (tV[i * d + row0 + r] - a * h0);
      const float u1 = b * (tV[i * d + row0 + r + 8] - a * h1);

      // S <- exp(g) S + k u^T, and the output off the state that leaves behind rather than off the
      // one it started from: o = q^T S_i is the same sum the chunk path spells as a decayed
      // q^T S_0 plus a row of the score matrix against u.
      float o0 = 0.0f;
      float o1 = 0.0f;
#pragma unroll
      for (int t = 0; t < nTilesD; ++t) {
        float k0 = tk[t * kTileN + c];
        float k1 = tk[t * kTileN + c + 1];
        float s0 = a * accState[t].x[0] + u0 * k0;
        float s1 = a * accState[t].x[1] + u0 * k1;
        float s2 = a * accState[t].x[2] + u1 * k0;
        float s3 = a * accState[t].x[3] + u1 * k1;
        accState[t].x[0] = s0;
        accState[t].x[1] = s1;
        accState[t].x[2] = s2;
        accState[t].x[3] = s3;

        float q0 = tq[t * kTileN + c];
        float q1 = tq[t * kTileN + c + 1];
        o0 += s0 * q0 + s1 * q1;
        o1 += s2 * q0 + s3 * q1;
      }
      o0 += __shfl_xor_sync(0xffffffff, o0, 1);
      o1 += __shfl_xor_sync(0xffffffff, o1, 1);
      o0 += __shfl_xor_sync(0xffffffff, o0, 2);
      o1 += __shfl_xor_sync(0xffffffff, o1, 2);

      // One lane of the four that hold a row has the whole sum; the output goes back through shared
      // memory so that it leaves coalesced.
      if ((lane & 3) == 0) {
        tO[i * d + row0 + r] = o0;
        tO[i * d + row0 + r + 8] = o1;
      }
    }
  }
  __syncthreads();

  for (int idx = tid; idx < len * d; idx += kThreads) {
    int i = idx / d;
    int c = idx - i * d;
    out[(static_cast<int64_t>(begin + i) * numVHead + head) * d + c] = __float2half(tO[idx]);
  }
}

/// The head dimension is a template parameter, not an argument: every accumulator array here is
/// indexed by a loop bound derived from it, and with a runtime bound ptxas puts them on the stack
/// -- 256 bytes a thread of it, which costs more than the shared memory round trips this path
/// exists to avoid.
template <int kD>
__global__ __launch_bounds__(kThreads) void gatedDeltaNetMmaKernel(
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
    int headRatio,
    int recurrentMax) {
#if __CUDA_ARCH__ >= 800
  constexpr int d = kD;
  const int head = blockIdx.x;
  const int seq = blockIdx.y;
  const int kHead = head / headRatio;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  const int begin = cuSeqlens[seq];
  const int end = cuSeqlens[seq + 1];
  if (begin >= end) return;

  constexpr int ld = d + kPad;
  constexpr int nb = d / kTileM;         // 16-wide blocks of the head dimension
  constexpr int nTilesD = d / kTileN;    // 8-wide tiles of it
  constexpr int vecPerRow = d / 8;       // 16-byte pieces of a row
  const bool ownsRows = warp < nb;       // this warp's 16 rows of the value dimension
  const int row0 = warp * kTileM;

  extern __shared__ char smemRaw[];
  float *sCum = reinterpret_cast<float *>(smemRaw);
  float *sCexp = sCum + kChunk;
  float *sClast = sCexp + kChunk;
  float *sBeta = sClast + kChunk;
  // Two key buffers, so the next chunk's keys can be on their way while this one still reads its
  // own. The queries and the values have one each: both fall out of use partway through a chunk,
  // which is early enough for the copy that refills them to be hidden.
  half *sKBuf[2];
  sKBuf[0] = reinterpret_cast<half *>(sBeta + kChunk);
  sKBuf[1] = sKBuf[0] + kChunk * ld;
  half *sQ = sKBuf[1] + kChunk * ld;
  half *sV = sQ + kChunk * ld;           // v on the way in, the output on the way out
  half *sScore = sV + kChunk * ld;
  half *sInv = sScore + kChunk * kScoreLd;

  // Issuing a chunk's copies happens in three places: once before the loop, and then for the next
  // chunk from inside this one, as each buffer falls out of use.
  auto issueCopies = [&](int start, bool wantK, bool wantQ, bool wantV, half *kDst) {
    if (start >= end) return;

    int chunkLen = min(kChunk, end - start);
    for (int idx = tid; idx < kChunk * vecPerRow; idx += kThreads) {
      int i = idx / vecPerRow;
      int m = (idx - i * vecPerRow) * 8;
      int bytes = (i < chunkLen) ? 16 : 0;
      int64_t kqOff = (static_cast<int64_t>(start + i) * numKHead + kHead) * d + m;
      if (wantK) copyAsync16(kDst + i * ld + m, k + kqOff, bytes);
      if (wantQ) copyAsync16(sQ + i * ld + m, q + kqOff, bytes);
      if (wantV) {
        copyAsync16(
            sV + i * ld + m,
            v + (static_cast<int64_t>(start + i) * numVHead + head) * d + m,
            bytes);
      }
    }
    copyAsyncCommit();
  };

  const int64_t stateOffset =
      (static_cast<int64_t>(stateSlots[seq]) * numVHead + head) * d * d;

  // The state, transposed, in registers for the whole sequence: this warp's 16 rows of the value
  // dimension by every column of the key dimension, as one m16n8 accumulator per eight columns.
  Acc accState[nTilesD];
  if (ownsRows) {
    const int r = accRow(lane);
    const int c = accCol(lane);
#pragma unroll
    for (int t = 0; t < nTilesD; ++t) {

      // state is (key dim, value dim); this holds its transpose, so a row here is a column there.
      const float *p = state + stateOffset + row0;
      accState[t].x[0] = p[static_cast<int64_t>(t * kTileN + c) * d + r];
      accState[t].x[1] = p[static_cast<int64_t>(t * kTileN + c + 1) * d + r];
      accState[t].x[2] = p[static_cast<int64_t>(t * kTileN + c) * d + r + 8];
      accState[t].x[3] = p[static_cast<int64_t>(t * kTileN + c + 1) * d + r + 8];
    }
  }

  // Short enough to step, or long enough to be worth a chunk. Both sides start from the state this
  // block just read and leave it in the same registers for the write-back below, so the branch is
  // over what happens to the sequence and nothing else.
  const int seqLen = end - begin;
  if (seqLen <= recurrentMax) {
    recurrentSequence<kD>(
        q, k, v, g, beta, out, accState, sCexp, sBeta, reinterpret_cast<float *>(sKBuf[0]),
        begin, seqLen, head, kHead, numKHead, numVHead, tid, lane, row0, ownsRows);
  } else {
    // The tiles of the two chunk-square matrices at or below the diagonal, as (row block, n tile).
    // The six above it are zeroed once and never computed; the inversion reads them as the zero
    // blocks of a block triangular matrix and nothing else looks at them.
    for (int idx = tid; idx < kChunk * kChunk / 8; idx += kThreads) {
      int i = idx / (kChunk / 8);
      int j = (idx - i * (kChunk / 8)) * 8;
      if (j > i) {
        const uint4 zero = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4 *>(sScore + i * kScoreLd + j) = zero;
        *reinterpret_cast<uint4 *>(sInv + i * kScoreLd + j) = zero;
      }
    }

    issueCopies(begin, true, true, true, sKBuf[0]);

    int parity = 0;
    for (int t0 = begin; t0 < end; t0 += kChunk, parity ^= 1) {
      const int len = min(kChunk, end - t0);
      const int next = t0 + kChunk;
      half *sK = sKBuf[parity];

      __syncthreads();

      // -- the decays, on one warp, entirely in shuffles.
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

      // -- K, Q and v, which the chunk before this one put on their way. Rows past the chunk's
      // length were zero filled by the copy itself.
      copyAsyncWait();
      __syncthreads();

      // -- A and the score matrix, K K^T and Q K^T decayed and masked. Both are 16 by 16 tiles at or
      // below the diagonal: twenty of them, over eight warps, each carrying two accumulators.
      for (int t = warp; t < 2 * 10; t += kWarps) {
        const bool isScore = t >= 10;
        const int tt = isScore ? t - 10 : t;
        // (0,0) (1,0) (1,1) (2,0) ... in row major order over the lower triangle.
        int rb = 0;
        int cb = tt;
        while (cb > rb) {
          cb -= rb + 1;
          ++rb;
        }

        Acc acc[2] = {};
        const half *a = isScore ? sQ : sK;
        for (int kk = 0; kk < nb; ++kk) {
          unsigned fa[4];
          unsigned fb[4];
          loadA(fa, a + rb * kTileM * ld + kk * kTileK, ld, lane);
          loadB(fb, sK + cb * kTileM * ld + kk * kTileK, ld, lane);
          mmaB(acc[0], acc[1], fa, fb);
        }

        const int r = accRow(lane);
        const int c = accCol(lane);
  #pragma unroll
        for (int half8 = 0; half8 < 2; ++half8) {
  #pragma unroll
          for (int e = 0; e < 4; ++e) {
            int i = rb * kTileM + r + (e >= 2 ? 8 : 0);
            int j = cb * kTileM + c + (e & 1) + half8 * 8;
            float val = 0.0f;
            if (i < len && j <= i) {
              // A difference of logs, which is the only safe form: either exponential on its own can
              // be far outside half's range where the ratio is not.
              float decay = expf(sCum[i] - sCum[j]);
              val = isScore ? decay * acc[half8].x[e] : sBeta[i] * decay * acc[half8].x[e];
            }
            if (!isScore && i == j) val = 1.0f;
            acc[half8].x[e] = val;
          }
        }
        storeTile(acc, (isScore ? sScore : sInv) + rb * kTileM * kScoreLd + cb * kTileM,
                  kScoreLd, lane, false);
      }
      __syncthreads();

      // -- the inverse of (I + A), on the first four warps; the other four walk straight past it into
      // the state products, which do not read it.
      if (warp < kKBlocks) {
        invertUnitLowerTile(sInv + warp * kTileM * kScoreLd + warp * kTileM, kScoreLd, lane);
        barrierInverseWarps();

        if (warp < 2) {
          int r = 2 * warp + 1;
          int c = 2 * warp;
          half *dst = sInv + r * kTileM * kScoreLd + c * kTileM;

          Acc acc[2];
          tileGemm(acc, dst, kScoreLd, sInv + c * kTileM * kScoreLd + c * kTileM, kScoreLd, 1, lane);
          storeTile(acc, dst, kScoreLd, lane, false);

          tileGemm(acc, sInv + r * kTileM * kScoreLd + r * kTileM, kScoreLd, dst, kScoreLd, 1, lane);
          storeTile(acc, dst, kScoreLd, lane, true);
        }
        barrierInverseWarps();

        const int invR = 2 + warp / 2;
        const int invC = warp % 2;
        Acc acc[2];
        tileGemm(
            acc,
            sInv + invR * kTileM * kScoreLd,
            kScoreLd,
            sInv + invC * kTileM,
            kScoreLd,
            2,
            lane);
        barrierInverseWarps();
        storeTile(acc, sInv + invR * kTileM * kScoreLd + invC * kTileM, kScoreLd, lane, false);
        barrierInverseWarps();

        tileGemm(
            acc,
            sInv + invR * kTileM * kScoreLd + 2 * kTileM,
            kScoreLd,
            sInv + 2 * kTileM * kScoreLd + invC * kTileM,
            kScoreLd,
            2,
            lane);
        barrierInverseWarps();
        storeTile(acc, sInv + invR * kTileM * kScoreLd + invC * kTileM, kScoreLd, lane, true);
      }

      // -- M K^T and M Q^T, the two passes over the state. The state is the A operand of both,
      // straight out of the accumulators it has been living in.
      Acc accU[kNTiles] = {};   // becomes the right hand side, then u
      Acc accO[kNTiles] = {};   // exp(c) M Q^T, which the output accumulates on top of
      if (ownsRows) {
        for (int kk = 0; kk < nb; ++kk) {
          unsigned fa[4];
          accToOperand(fa, accState[2 * kk], accState[2 * kk + 1]);

          unsigned fk[4];
          unsigned fq[4];
          loadB(fk, sK + kk * kTileK, ld, lane);
          loadB(fq, sQ + kk * kTileK, ld, lane);
  #pragma unroll
          for (int t = 0; t < kNTiles; t += 2) {
            // Sixteen tokens a load, and the B operands for both of them come out of the same
            // ldmatrix.
            if (t != 0) {
              loadB(fk, sK + (t / 2) * kTileM * ld + kk * kTileK, ld, lane);
              loadB(fq, sQ + (t / 2) * kTileM * ld + kk * kTileK, ld, lane);
            }
            mmaB(accU[t], accU[t + 1], fa, fk);
            mmaB(accO[t], accO[t + 1], fa, fq);
          }
        }
      }

      // -- the right hand side, beta (v - exp(c) K S_0), and the decay on the output's first term.
      // Both are a multiply on a register here: the token is the accumulator's column.
      if (ownsRows) {
        const int r = accRow(lane);
        const int c = accCol(lane);
  #pragma unroll
        for (int t = 0; t < kNTiles; ++t) {
  #pragma unroll
          for (int e = 0; e < 4; ++e) {
            int tok = t * kTileN + c + (e & 1);
            int dim = row0 + r + (e >= 2 ? 8 : 0);
            float vv = __half2float(sV[tok * ld + dim]);
            float b = sBeta[tok];
            float decay = sCexp[tok];
            accU[t].x[e] = (tok < len) ? b * (vv - decay * accU[t].x[e]) : 0.0f;
            accO[t].x[e] *= decay;
          }
        }
      }
      __syncthreads();

      // The state products were the last reader of the queries, and the keys the next chunk wants go
      // to the buffer this one is not using, so both copies start here -- against the solve, the
      // output and the state update below, rather than in front of the next chunk. Leaving the values
      // in flight further than this, with a wait_group of one at the top of the chunk, was tried and
      // is 2% slower: it delays these two by more than it saves on those.
      issueCopies(next, true, true, false, sKBuf[parity ^ 1]);

      // -- u^T = rhs^T (I + A)^-T, and the output. Both read what the phase before them left in
      // registers as their A operand.
      if (ownsRows) {
        // The right hand side becomes A operands before it is overwritten, which is cheaper than
        // keeping a second set of accumulators alive to copy from.
        unsigned faRhs[kKBlocks][4];
  #pragma unroll
        for (int kk = 0; kk < kKBlocks; ++kk) {
          accToOperand(faRhs[kk], accU[2 * kk], accU[2 * kk + 1]);
        }
  #pragma unroll
        for (int t = 0; t < kNTiles; ++t) accU[t] = {};

  #pragma unroll
        for (int kk = 0; kk < kKBlocks; ++kk) {
  #pragma unroll
          for (int t = 0; t < kNTiles; t += 2) {
            // The inverse is lower triangular: a pair of token tiles only reads the blocks of the
            // contraction at or before the last token in it.
            if (kk * kTileK > (t + 1) * kTileN + kTileN - 1) continue;

            unsigned fb[4];
            loadB(fb, sInv + (t / 2) * kTileM * kScoreLd + kk * kTileK, kScoreLd, lane);
            mmaB(accU[t], accU[t + 1], faRhs[kk], fb);
          }
        }

        unsigned faU[kKBlocks][4];
  #pragma unroll
        for (int kk = 0; kk < kKBlocks; ++kk) {
          accToOperand(faU[kk], accU[2 * kk], accU[2 * kk + 1]);
        }
  #pragma unroll
        for (int kk = 0; kk < kKBlocks; ++kk) {
  #pragma unroll
          for (int t = 0; t < kNTiles; t += 2) {
            if (kk * kTileK > (t + 1) * kTileN + kTileN - 1) continue;

            unsigned fb[4];
            loadB(fb, sScore + (t / 2) * kTileM * kScoreLd + kk * kTileK, kScoreLd, lane);
            mmaB(accO[t], accO[t + 1], faU[kk], fb);
          }
        }
      }
      // -- the output, which comes out with the token on the column and has to leave with it on the
      // row. It goes back through v's buffer, which is dead now, and out coalesced.
      if (ownsRows) {
        const int r = accRow(lane);
        const int c = accCol(lane);
  #pragma unroll
        for (int t = 0; t < kNTiles; ++t) {
  #pragma unroll
          for (int e = 0; e < 4; ++e) {
            int tok = t * kTileN + c + (e & 1);
            int dim = row0 + r + (e >= 2 ? 8 : 0);
            sV[tok * ld + dim] = __float2half(accO[t].x[e]);
          }
        }
      }
      __syncthreads();

      for (int idx = tid; idx < len * vecPerRow; idx += kThreads) {
        int i = idx / vecPerRow;
        int m = (idx - i * vecPerRow) * 8;
        *reinterpret_cast<uint4 *>(out + (static_cast<int64_t>(t0 + i) * numVHead + head) * d + m) =
            *reinterpret_cast<const uint4 *>(sV + i * ld + m);
      }
      __syncthreads();

      // The values' buffer only falls free here, since the output is staged in it.
      issueCopies(next, false, false, true, nullptr);

      // -- the state. The decay of the chunk's tail goes onto u's columns, again a register away, and
      // the product accumulates straight into the state's own fragments.
      if (ownsRows) {
        const int c = accCol(lane);
        const float gamma = sCexp[len - 1];
  #pragma unroll
        for (int t = 0; t < nTilesD; ++t) {
  #pragma unroll
          for (int e = 0; e < 4; ++e) accState[t].x[e] *= gamma;
        }

  #pragma unroll
        for (int t = 0; t < kNTiles; ++t) {
  #pragma unroll
          for (int e = 0; e < 4; ++e) {
            int tok = t * kTileN + c + (e & 1);
            accU[t].x[e] *= sClast[tok];
          }
        }

        for (int kk = 0; kk < kKBlocks; ++kk) {
          unsigned fa[4];
          accToOperand(fa, accU[2 * kk], accU[2 * kk + 1]);
  #pragma unroll
          for (int t = 0; t < nTilesD; t += 2) {
            unsigned fb[4];
            loadBTrans(fb, sK + kk * kTileK * ld + (t / 2) * kTileM, ld, lane);
            mmaBTrans(accState[t], accState[t + 1], fa, fb);
          }
        }
      }
    }
  }

  if (ownsRows) {
    const int r = accRow(lane);
    const int c = accCol(lane);
    float *p = state + stateOffset + row0;
#pragma unroll
    for (int t = 0; t < nTilesD; ++t) {
      p[static_cast<int64_t>(t * kTileN + c) * d + r] = accState[t].x[0];
      p[static_cast<int64_t>(t * kTileN + c + 1) * d + r] = accState[t].x[1];
      p[static_cast<int64_t>(t * kTileN + c) * d + r + 8] = accState[t].x[2];
      p[static_cast<int64_t>(t * kTileN + c + 1) * d + r + 8] = accState[t].x[3];
    }
  }
#endif
}

size_t sharedBytes(int d) {
  int ld = d + kPad;
  size_t floats = 4 * kChunk;
  size_t halves = static_cast<size_t>(kChunk) * ld * 4 +        // two key buffers, sQ, sV
                  static_cast<size_t>(kChunk) * kScoreLd * 2;   // sScore, sInv
  return floats * sizeof(float) + halves * sizeof(half);
}

}  // namespace

bool fits(int headDim, int *smemOut) {
  // Only the head dimensions the kernel is instantiated for; see the template above.
  if (headDim != 32 && headDim != 64 && headDim != 128) return false;

  int major = 0;
  int minor = 0;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
  if (major * 10 + minor < kMinArch) return false;

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
    int numSeq,
    int recurrentMax) {
  int smem = 0;
  CHECK(fits(headDim, &smem))
      << "the mma gatedDeltaNetPrefill needs a head dimension that is a multiple of " << kTileM
      << " and at most " << kMaxHeadDim << ", got " << headDim;

  auto launch = [&](auto kernel) {
    LL_CHECK_CUDA_STATUS(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<dim3(numVHead, numSeq), kThreads, smem>>>(
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
        numVHead / numKHead,
        std::min(recurrentMax, kMaxRecurrentLen));
  };

  switch (headDim) {
    case 32:
      launch(gatedDeltaNetMmaKernel<32>);
      break;
    case 64:
      launch(gatedDeltaNetMmaKernel<64>);
      break;
    case 128:
      launch(gatedDeltaNetMmaKernel<128>);
      break;
    default:
      NOT_IMPL();
  }
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace gdnmma
}  // namespace cuda
}  // namespace op
}  // namespace fl
