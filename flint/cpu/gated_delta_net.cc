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

#include "flint/cpu/gated_delta_net.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/tensor.h"
#include "flint/cpu/triangular_solve.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

// One (sequence, value head) pair carried from its incoming state to its outgoing one.
//
// Within a chunk the recurrence
//   S_i = a_i S_{i-1} + k_i u_i^T,   u_i = beta_i (v_i - a_i S_{i-1}^T k_i),   a_i = exp(g_i)
// unrolls to
//   S_i = exp(c_i) S_0 + sum_{j<=i} exp(c_i - c_j) k_j u_j^T,   c_i = sum_{j<=i} g_j
// so substituting it back into the definition of u_i leaves the u's as the solution of
//   (I + A) U = beta * V - diag(beta exp(c)) K S_0,   A_ij = beta_i exp(c_i - c_j) k_i . k_j, j < i
// which is the unit lower triangular system the chunk solves. Splitting the right hand side in two
// keeps S_0 out of the solve: U comes from beta * V and W from diag(beta exp(c)) K, both
// independent of the state, and the state only enters afterwards as u = U - W S_0. Every decay
// that appears is exp of a sum of g's over a window, and g is at most 0, so no factor here can
// exceed 1.
void runHead(
    const TensorAccessor<const float, 3> &q,
    const TensorAccessor<const float, 3> &k,
    const TensorAccessor<const float, 3> &v,
    const TensorAccessor<const float, 2> &g,
    const TensorAccessor<const float, 2> &beta,
    TensorAccessor<float, 2> state,
    TensorAccessor<float, 3> o,
    int begin,
    int end,
    int head,
    int kHead,
    int headDim) {
  constexpr int C = kGatedDeltaNetChunkSize;
  int d = headDim;

  std::vector<float> a(static_cast<size_t>(C) * C);
  std::vector<float> rhs(static_cast<size_t>(C) * 2 * d);
  std::vector<float> uhat(static_cast<size_t>(C) * d);
  std::vector<float> cum(C);
  std::vector<float> cexp(C);
  std::vector<float> clast(C);

  for (int t0 = begin; t0 < end; t0 += C) {
    int len = std::min(C, end - t0);

    // The cumulative log decay of the chunk, kept in the log domain: every decay the chunk needs
    // spans a window of it, and taking the difference before the exponential keeps a long run of
    // strongly decaying steps from underflowing one exponential and dividing by it.
    float cumulative = 0.0f;
    for (int i = 0; i < len; ++i) {
      cumulative += g[t0 + i][head];
      cum[i] = cumulative;
      cexp[i] = expf(cumulative);
    }
    for (int i = 0; i < len; ++i) clast[i] = expf(cumulative - cum[i]);

    for (int i = 0; i < len; ++i) {
      float bi = beta[t0 + i][head];
      float *arow = a.data() + static_cast<size_t>(i) * C;
      for (int j = 0; j < i; ++j) {
        float dot = 0.0f;
        for (int c = 0; c < d; ++c) dot += k[t0 + i][kHead][c] * k[t0 + j][kHead][c];

        // exp(c_i - c_j) for j < i, which is exp of the g's from j + 1 up to i.
        arow[j] = bi * expf(cum[i] - cum[j]) * dot;
      }
      arow[i] = 1.0f;
      for (int j = i + 1; j < len; ++j) arow[j] = 0.0f;

      float *r = rhs.data() + static_cast<size_t>(i) * 2 * d;
      for (int c = 0; c < d; ++c) r[c] = bi * v[t0 + i][head][c];
      for (int c = 0; c < d; ++c) r[d + c] = bi * cexp[i] * k[t0 + i][kHead][c];
    }

    triangularSolveRowMajor(a.data(), C, rhs.data(), 2 * d, len, 2 * d);

    // u = U - W S_0, the chunk's writes once the incoming state is accounted for.
    for (int i = 0; i < len; ++i) {
      const float *r = rhs.data() + static_cast<size_t>(i) * 2 * d;
      float *u = uhat.data() + static_cast<size_t>(i) * d;
      for (int c = 0; c < d; ++c) u[c] = r[c];
      for (int m = 0; m < d; ++m) {
        float w = r[d + m];
        if (w == 0.0f) continue;
        for (int c = 0; c < d; ++c) u[c] -= w * state[m][c];
      }
    }

    // o_i = exp(c_i) q_i^T S_0 + sum_{j<=i} exp(c_i - c_j) (q_i . k_j) u_j
    for (int i = 0; i < len; ++i) {
      for (int c = 0; c < d; ++c) o[t0 + i][head][c] = 0.0f;
      for (int m = 0; m < d; ++m) {
        float qm = cexp[i] * q[t0 + i][kHead][m];
        for (int c = 0; c < d; ++c) o[t0 + i][head][c] += qm * state[m][c];
      }

      for (int j = 0; j <= i; ++j) {
        float dot = 0.0f;
        for (int m = 0; m < d; ++m) dot += q[t0 + i][kHead][m] * k[t0 + j][kHead][m];

        float s = expf(cum[i] - cum[j]) * dot;
        const float *u = uhat.data() + static_cast<size_t>(j) * d;
        for (int c = 0; c < d; ++c) o[t0 + i][head][c] += s * u[c];
      }
    }

    // S <- exp(c_last) S_0 + sum_j exp(c_last - c_j) k_j u_j^T
    float gamma = cexp[len - 1];
    for (int m = 0; m < d; ++m) {
      for (int c = 0; c < d; ++c) state[m][c] *= gamma;
    }
    for (int j = 0; j < len; ++j) {
      const float *u = uhat.data() + static_cast<size_t>(j) * d;
      for (int m = 0; m < d; ++m) {
        float km = clast[j] * k[t0 + j][kHead][m];
        for (int c = 0; c < d; ++c) state[m][c] += km * u[c];
      }
    }
  }
}

}  // namespace

Tensor gatedDeltaNetPrefill(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &g,
    const Tensor &beta,
    const Tensor &cuSeqlens,
    const Tensor &stateSlots,
    Tensor &state) {
  CHECK(q.getDType() == DType::kFloat && k.getDType() == DType::kFloat);
  CHECK(v.getDType() == DType::kFloat && state.getDType() == DType::kFloat);
  CHECK(g.getDType() == DType::kFloat && beta.getDType() == DType::kFloat);
  CHECK(cuSeqlens.getDType() == DType::kInt32 && stateSlots.getDType() == DType::kInt32);
  CHECK(q.getDim() == 3 && k.getDim() == 3 && v.getDim() == 3);
  CHECK(g.getDim() == 2 && beta.getDim() == 2 && cuSeqlens.getDim() == 1);
  CHECK(stateSlots.getDim() == 1);
  CHECK(state.getDim() == 4);

  int numTokens = q.getShape(0);
  int numKHead = q.getShape(1);
  int headDim = q.getShape(2);
  int numVHead = v.getShape(1);
  int numSeq = cuSeqlens.getShape(0) - 1;
  int numStateSlot = state.getShape(0);

  CHECK(k.getShape(0) == numTokens && k.getShape(1) == numKHead && k.getShape(2) == headDim);
  CHECK(v.getShape(0) == numTokens && v.getShape(2) == headDim);
  CHECK(g.getShape(0) == numTokens && g.getShape(1) == numVHead);
  CHECK(beta.getShape(0) == numTokens && beta.getShape(1) == numVHead);
  CHECK(numVHead % numKHead == 0) << "the value heads must be a multiple of the key heads";
  CHECK(stateSlots.getShape(0) == numSeq) << "one state slot per sequence";
  CHECK(state.getShape(0) >= numSeq && state.getShape(1) == numVHead);
  CHECK(state.getShape(2) == headDim && state.getShape(3) == headDim);

  int headRatio = numVHead / numKHead;
  Tensor o = tensor({numTokens, numVHead, headDim}, DType::kFloat);

  TensorAccessor<const float, 3> aQ(q);
  TensorAccessor<const float, 3> aK(k);
  TensorAccessor<const float, 3> aV(v);
  TensorAccessor<const float, 2> aG(g);
  TensorAccessor<const float, 2> aBeta(beta);
  TensorAccessor<const IntType, 1> aCuSeqlens(cuSeqlens);
  TensorAccessor<const IntType, 1> aStateSlots(stateSlots);
  TensorAccessor<float, 4> aState(state);
  TensorAccessor<float, 3> aO(o);

  int numJobs = numSeq * numVHead;
#pragma omp parallel for schedule(dynamic, 1)
  for (int job = 0; job < numJobs; ++job) {
    int seq = job / numVHead;
    int head = job - seq * numVHead;
    int begin = aCuSeqlens[seq];
    int end = aCuSeqlens[seq + 1];
    CHECK(begin >= 0 && begin <= end && end <= numTokens);
    if (begin == end) continue;

    // Which slot of the pool this sequence's state lives in. It has nothing to do with where the
    // sequence sits in the batch, which is the point of the mapping.
    int slot = aStateSlots[seq];
    CHECK(slot >= 0 && slot < numStateSlot) << "state slot out of the pool";

    runHead(aQ, aK, aV, aG, aBeta, aState[slot][head], aO, begin, end, head, head / headRatio,
            headDim);
  }

  return o;
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
