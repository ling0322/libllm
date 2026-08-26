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

#include "flint/cuda/triangular_solve.h"

#include <cuda_fp16.h>

#include <limits>

#include "flint/cuda/common.h"
#include "flint/cuda/copy.h"

namespace fl {
namespace op {
namespace cuda {

namespace {

__device__ inline float loadElem(float v) {
  return v;
}
__device__ inline float loadElem(half v) {
  return __half2float(v);
}
__device__ inline void storeElem(float *p, float v) {
  *p = v;
}
__device__ inline void storeElem(half *p, float v) {
  *p = __float2half(v);
}

// A thread owns every row of its own columns, so the substitution inside a row block stays in that
// thread's registers and the kernel needs no barrier past the initial load of L. That is what fixes
// the tile shape: kRowBlock rows are live in registers at once, so kRowBlock * kColsPerThread is
// the accumulator count, and the reduction over the rows already solved reads kRowBlock values of L
// against kColsPerThread values of X.
//
// At the sizes the delta rule solves at -- N of 64 against a few hundred columns -- this is a
// bandwidth bound problem, not an arithmetic one: it moves N * M elements to do N^2/2 * M
// multiply-adds, which is 4 FMA per byte where the hardware wants nearer 27. So the width the right
// hand side is stored at matters more than the tiling does, which is why the element type is a
// template parameter and the caller gets to pick.
constexpr int kRowBlock = 32;
constexpr int kColsPerThread = 2;
constexpr int kNumThreads = 128;

// The largest N whose L fits in the shared memory tile. Wider systems take the fallback kernel.
constexpr int kMaxN = 64;

constexpr int kSimpleNumThreads = 256;

}  // namespace

/// Forward substitution over row blocks. Each block owns one system and a tile of
/// kNumThreads * TN of its columns; each thread owns TN columns outright, from the first row to
/// the last, so the rows it reads back are always rows it wrote itself.
template<int BN, int TN, int MAXN, int NUM_THREADS, typename XT>
__global__ void triangularSolveTiledKernel(
    const float *__restrict__ l,
    XT *__restrict__ x,
    int n,
    int m) {
  // lt[j][i] is L[i][j]. A row block reads one column of L across all of its rows, so holding L
  // transposed makes that a contiguous shared read that every thread in the warp broadcasts.
  __shared__ float lt[MAXN][MAXN];

  const float *lp = l + static_cast<int64_t>(blockIdx.y) * n * n;
  XT *xp = x + static_cast<int64_t>(blockIdx.y) * n * m;

  for (int idx = threadIdx.x; idx < MAXN * MAXN; idx += NUM_THREADS) {
    lt[idx / MAXN][idx % MAXN] = 0.0f;
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < n * n; idx += NUM_THREADS) {
    int r = idx / n;
    lt[idx - r * n][r] = lp[idx];
  }
  __syncthreads();

  const int col0 = (blockIdx.x * NUM_THREADS + threadIdx.x) * TN;
  if (col0 >= m) return;

  float acc[BN][TN];
  for (int i0 = 0; i0 < n; i0 += BN) {
    int rows = min(BN, n - i0);

#pragma unroll
    for (int a = 0; a < BN; ++a) {
#pragma unroll
      for (int b = 0; b < TN; ++b) {
        int col = col0 + b;
        acc[a][b] = (a < rows && col < m)
                        ? loadElem(xp[static_cast<int64_t>(i0 + a) * m + col])
                        : 0.0f;
      }
    }

    // The rows solved by the earlier row blocks. This thread wrote them, so reading them back
    // takes no barrier.
    for (int j = 0; j < i0; ++j) {
      float xj[TN];
#pragma unroll
      for (int b = 0; b < TN; ++b) {
        int col = col0 + b;
        xj[b] = (col < m) ? loadElem(xp[static_cast<int64_t>(j) * m + col]) : 0.0f;
      }

      const float *lcol = &lt[j][i0];
#pragma unroll
      for (int a = 0; a < BN; ++a) {
        float lv = lcol[a];
#pragma unroll
        for (int b = 0; b < TN; ++b) {
          acc[a][b] -= lv * xj[b];
        }
      }
    }

    // The row block's own triangle, entirely in registers.
#pragma unroll
    for (int r = 0; r < BN; ++r) {
      if (r >= rows) continue;

      float inv = 1.0f / lt[i0 + r][i0 + r];
#pragma unroll
      for (int b = 0; b < TN; ++b) {
        acc[r][b] *= inv;
        int col = col0 + b;
        if (col < m) storeElem(&xp[static_cast<int64_t>(i0 + r) * m + col], acc[r][b]);
      }

      const float *lcol = &lt[i0 + r][i0];
#pragma unroll
      for (int a = r + 1; a < BN; ++a) {
        float lv = lcol[a];
#pragma unroll
        for (int b = 0; b < TN; ++b) {
          acc[a][b] -= lv * acc[r][b];
        }
      }
    }
  }
}

/// The fallback for systems too wide for the shared memory tile: one thread per column, walking
/// the rows one at a time and reading the rows it already solved back from global memory.
template<typename XT>
__global__ void triangularSolveSimpleKernel(
    const float *__restrict__ l,
    XT *__restrict__ x,
    int n,
    int m) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= m) return;

  const float *lp = l + static_cast<int64_t>(blockIdx.y) * n * n;
  XT *xp = x + static_cast<int64_t>(blockIdx.y) * n * m;

  for (int i = 0; i < n; ++i) {
    const float *li = lp + static_cast<int64_t>(i) * n;
    float acc = loadElem(xp[static_cast<int64_t>(i) * m + col]);
    for (int j = 0; j < i; ++j) {
      acc -= li[j] * loadElem(xp[static_cast<int64_t>(j) * m + col]);
    }
    storeElem(&xp[static_cast<int64_t>(i) * m + col], acc / li[i]);
  }
}

void triangularSolveInplace(const Tensor &l, Tensor &x) {
  CHECK(l.getDevice().getType() == Device::kCuda);
  CHECK(x.getDevice().getType() == Device::kCuda);
  CHECK(l.getDType() == DType::kFloat) << "the coefficient matrices must be <float>";
  CHECK(x.getDType() == DType::kFloat || x.getDType() == DType::kFloat16);
  LL_CHECK_CONTIGUOUS(l);
  LL_CHECK_CONTIGUOUS(x);
  CHECK(l.getDim() >= 2 && x.getDim() == l.getDim());

  int n = l.getShape(-1);
  CHECK(l.getShape(-2) == n) << "the coefficient matrices of triangularSolve must be square";
  CHECK(x.getShape(-2) == n);
  int m = x.getShape(-1);

  int64_t numSystems = 1;
  for (int d = 0; d < l.getDim() - 2; ++d) {
    CHECK(l.getShape(d) == x.getShape(d));
    numSystems *= l.getShape(d);
  }
  CHECK(numSystems <= std::numeric_limits<int>::max());
  if (numSystems == 0 || n == 0 || m == 0) return;

  const float *lp = getDataPtrCuda<float>(l);
  constexpr int colsPerBlock = kNumThreads * kColsPerThread;
  dim3 tiledGrid((m + colsPerBlock - 1) / colsPerBlock, static_cast<int>(numSystems));
  dim3 simpleGrid((m + kSimpleNumThreads - 1) / kSimpleNumThreads, static_cast<int>(numSystems));

  // The substitution always accumulates in float; the element type only says how the right hand
  // side is stored. The gated DeltaNet prefill keeps its two right hand sides in half, where the
  // whole batch is several hundred megabytes and every one of them is read again by the scan.
  if (x.getDType() == DType::kFloat) {
    float *xp = getDataPtrCuda<float>(x);
    if (n <= kMaxN) {
      triangularSolveTiledKernel<kRowBlock, kColsPerThread, kMaxN, kNumThreads, float>
          <<<tiledGrid, kNumThreads>>>(lp, xp, n, m);
    } else {
      triangularSolveSimpleKernel<float><<<simpleGrid, kSimpleNumThreads>>>(lp, xp, n, m);
    }
  } else {
    half *xp = getDataPtrCuda<half>(x);
    if (n <= kMaxN) {
      triangularSolveTiledKernel<kRowBlock, kColsPerThread, kMaxN, kNumThreads, half>
          <<<tiledGrid, kNumThreads>>>(lp, xp, n, m);
    } else {
      triangularSolveSimpleKernel<half><<<simpleGrid, kSimpleNumThreads>>>(lp, xp, n, m);
    }
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
