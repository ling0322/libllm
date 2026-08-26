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

// The inputs of one prefill, laid out the way the operator takes them. The model L2 normalises the
// queries and keys before the linear attention, and the test does the same: a unit key is what
// makes the delta rule a contraction, so it is the regime the operator has to be right in.
// The regimes the decays and the write strengths are drawn from. The default is what a trained
// model looks like; the others are the ends the operator has to survive rather than the middle.
enum class Regime {
  kDefault,
  kNoDecay,      // every exp(c_i - c_j) is exactly one
  kStrongDecay,  // exp(c) underflows inside a chunk
  kBetaEnds,     // beta is zero on half the tokens and two on the other half
};

// How the state slots are handed out. Reversed is the usual case; scattered adds a pool twice the
// size it needs and a mapping that is neither monotonic nor contiguous.
enum class SlotPolicy {
  kReversed,
  kScattered,
};

struct Inputs {
  std::vector<int32_t> cuSeqlens;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> g;
  std::vector<float> beta;
  std::vector<float> state;

  int numTokens = 0;
  int numKHead = 0;
  int numVHead = 0;
  int headDim = 0;
  int numSeq = 0;
  SlotPolicy slotPolicy = SlotPolicy::kReversed;
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

Inputs makeInputs(
    const std::vector<int> &seqlens,
    int numKHead,
    int headRatio,
    int headDim,
    bool nonZeroState,
    uint64_t seed,
    Regime regime = Regime::kDefault,
    SlotPolicy slotPolicy = SlotPolicy::kReversed) {
  Lcg lcg(seed);
  Inputs in;
  in.slotPolicy = slotPolicy;
  in.numKHead = numKHead;
  in.numVHead = numKHead * headRatio;
  in.headDim = headDim;
  in.numSeq = static_cast<int>(seqlens.size());

  in.cuSeqlens.push_back(0);
  for (int len : seqlens) in.cuSeqlens.push_back(in.cuSeqlens.back() + len);
  in.numTokens = in.cuSeqlens.back();

  in.q.resize(static_cast<size_t>(in.numTokens) * numKHead * headDim);
  in.k.resize(in.q.size());
  in.v.resize(static_cast<size_t>(in.numTokens) * in.numVHead * headDim);
  in.g.resize(static_cast<size_t>(in.numTokens) * in.numVHead);
  in.beta.resize(in.g.size());
  in.state.resize(
      static_cast<size_t>(in.numSeq) * in.numVHead * headDim * headDim);

  for (float &x : in.q) x = lcg.next();
  for (float &x : in.k) x = lcg.next();
  for (float &x : in.v) x = lcg.next();
  normalizeRows(in.q, in.numTokens * numKHead, headDim);
  normalizeRows(in.k, in.numTokens * numKHead, headDim);

  // g is a log decay, so at most 0, and beta is a write strength in (0, 2).
  switch (regime) {
    case Regime::kNoDecay:
      for (float &x : in.g) x = 0.0f;
      for (float &x : in.beta) x = lcg.next() + 0.5f;
      break;
    case Regime::kStrongDecay:
      for (float &x : in.g) x = -(lcg.next() + 0.5f) * 6.0f;
      for (float &x : in.beta) x = lcg.next() + 0.5f;
      break;
    case Regime::kBetaEnds:
      for (float &x : in.g) x = -(lcg.next() + 0.5f) * 0.5f;
      for (size_t i = 0; i < in.beta.size(); ++i) in.beta[i] = (i % 2 == 0) ? 0.0f : 2.0f;
      break;
    default:
      for (float &x : in.g) x = -(lcg.next() + 0.5f) * 0.5f;
      for (float &x : in.beta) x = lcg.next() + 0.5f;
      break;
  }
  if (nonZeroState) {
    for (float &x : in.state) x = lcg.next() * 0.2f;
  }

  return in;
}

// The recurrence spelled out one token at a time, which is the definition the chunked operator has
// to agree with.
void referenceRecurrence(const Inputs &in, std::vector<float> *o, std::vector<float> *state) {
  int d = in.headDim;
  int headRatio = in.numVHead / in.numKHead;
  o->assign(static_cast<size_t>(in.numTokens) * in.numVHead * d, 0.0f);
  *state = in.state;

  std::vector<float> pred(d);
  for (int s = 0; s < in.numSeq; ++s) {
    for (int h = 0; h < in.numVHead; ++h) {
      int kh = h / headRatio;
      float *sm = state->data() + (static_cast<size_t>(s) * in.numVHead + h) * d * d;

      for (int t = in.cuSeqlens[s]; t < in.cuSeqlens[s + 1]; ++t) {
        const float *qt = in.q.data() + (static_cast<size_t>(t) * in.numKHead + kh) * d;
        const float *kt = in.k.data() + (static_cast<size_t>(t) * in.numKHead + kh) * d;
        const float *vt = in.v.data() + (static_cast<size_t>(t) * in.numVHead + h) * d;
        float a = std::exp(in.g[static_cast<size_t>(t) * in.numVHead + h]);
        float b = in.beta[static_cast<size_t>(t) * in.numVHead + h];

        // What the decayed state already predicts for this key.
        for (int c = 0; c < d; ++c) pred[c] = 0.0f;
        for (int m = 0; m < d; ++m) {
          for (int c = 0; c < d; ++c) pred[c] += a * sm[m * d + c] * kt[m];
        }

        for (int m = 0; m < d; ++m) {
          for (int c = 0; c < d; ++c) {
            sm[m * d + c] = a * sm[m * d + c] + kt[m] * b * (vt[c] - pred[c]);
          }
        }

        float *ot = o->data() + (static_cast<size_t>(t) * in.numVHead + h) * d;
        for (int m = 0; m < d; ++m) {
          for (int c = 0; c < d; ++c) ot[c] += qt[m] * sm[m * d + c];
        }
      }
    }
  }
}

// The state pool the tests hand the operator is deliberately not in batch order: it holds one
// slot more than the batch needs and sequence i owns slot numSeq - i, so an implementation that
// indexed the pool by position in the batch instead of through the mapping would read and write
// the wrong slots, and the spare slot 0 would not survive the call.
int slotOf(int seq, int numSeq, SlotPolicy policy) {
  if (policy == SlotPolicy::kScattered) return 1 + (seq * 7 + 3) % (2 * numSeq);
  return numSeq - seq;
}

int poolSlots(int numSeq, SlotPolicy policy) {
  return (policy == SlotPolicy::kScattered) ? 2 * numSeq + 1 : numSeq + 1;
}

// What the slot no sequence maps to holds, going in and coming out.
constexpr float kUnusedSlot = 7.0f;

// Lay one state per sequence out into the pool the mapping describes.
std::vector<float> toPool(const std::vector<float> &perSeq, const Inputs &in) {
  size_t slotSize = static_cast<size_t>(in.numVHead) * in.headDim * in.headDim;
  std::vector<float> pool(poolSlots(in.numSeq, in.slotPolicy) * slotSize, kUnusedSlot);
  for (int seq = 0; seq < in.numSeq; ++seq) {
    std::copy(
        perSeq.begin() + seq * slotSize,
        perSeq.begin() + (seq + 1) * slotSize,
        pool.begin() + slotOf(seq, in.numSeq, in.slotPolicy) * slotSize);
  }

  return pool;
}

struct Tensors {
  Tensor q, k, v, g, beta, cuSeqlens, stateSlots, state;
};

Tensors toTensors(const Inputs &in) {
  Tensors t;
  t.q = Tensor::create<float>({in.numTokens, in.numKHead, in.headDim}, lut::makeConstSpan(in.q));
  t.k = Tensor::create<float>({in.numTokens, in.numKHead, in.headDim}, lut::makeConstSpan(in.k));
  t.v = Tensor::create<float>({in.numTokens, in.numVHead, in.headDim}, lut::makeConstSpan(in.v));
  t.g = Tensor::create<float>({in.numTokens, in.numVHead}, lut::makeConstSpan(in.g));
  t.beta = Tensor::create<float>({in.numTokens, in.numVHead}, lut::makeConstSpan(in.beta));
  t.cuSeqlens = Tensor::create<int32_t>({in.numSeq + 1}, lut::makeConstSpan(in.cuSeqlens));
  std::vector<int32_t> slots;
  for (int seq = 0; seq < in.numSeq; ++seq) {
    slots.push_back(slotOf(seq, in.numSeq, in.slotPolicy));
  }
  t.stateSlots = Tensor::create<int32_t>({in.numSeq}, lut::makeConstSpan(slots));

  std::vector<float> pool = toPool(in.state, in);
  t.state = Tensor::create<float>(
      {poolSlots(in.numSeq, in.slotPolicy), in.numVHead, in.headDim, in.headDim},
      lut::makeConstSpan(pool));

  return t;
}

void checkAgainstRecurrence(const Inputs &in, float tolerance) {
  std::vector<float> expectedO;
  std::vector<float> expectedState;
  referenceRecurrence(in, &expectedO, &expectedState);

  Tensors t = toTensors(in);
  Tensor o = F::gatedDeltaNetPrefill(
      t.q,
      t.k,
      t.v,
      t.g,
      t.beta,
      t.cuSeqlens,
      t.stateSlots,
      t.state);

  CATCH_REQUIRE(o.getShape() == std::vector<int>{in.numTokens, in.numVHead, in.headDim});
  CATCH_REQUIRE(F::allClose(
      o,
      Tensor::create<float>(
          {in.numTokens, in.numVHead, in.headDim},
          lut::makeConstSpan(expectedO)),
      tolerance,
      tolerance));
  // Compared as a whole pool, so the spare slot's kUnusedSlot has to have survived along with
  // every sequence's state having landed in the slot the mapping named.
  std::vector<float> expectedPool = toPool(expectedState, in);
  CATCH_REQUIRE(F::allClose(
      t.state,
      Tensor::create<float>(
          {poolSlots(in.numSeq, in.slotPolicy), in.numVHead, in.headDim, in.headDim},
          lut::makeConstSpan(expectedPool)),
      tolerance,
      tolerance));
}

}  // namespace

CATCH_TEST_CASE("gatedDeltaNetPrefill matches the recurrence", "[op][cpu][gated_delta_net]") {
  // Inside one chunk, exactly one chunk, and a chunk plus a partial one.
  for (int len : {1, 5, 63, 64, 65, 150}) {
    CATCH_INFO("len = " << len);
    checkAgainstRecurrence(makeInputs({len}, 1, 1, 8, false, 0x100 + len), 1e-3f);
  }
}

CATCH_TEST_CASE("gatedDeltaNetPrefill carries an incoming state", "[op][cpu][gated_delta_net]") {
  // A state that is already non-zero is what tells the two halves of the right hand side apart:
  // it only reaches the output through W, which a zero state hides entirely.
  checkAgainstRecurrence(makeInputs({100}, 1, 1, 16, true, 0x200), 1e-3f);
}

CATCH_TEST_CASE("gatedDeltaNetPrefill takes a varlen batch", "[op][cpu][gated_delta_net]") {
  checkAgainstRecurrence(makeInputs({5, 64, 100, 1}, 2, 1, 16, true, 0x300), 2e-3f);
}

CATCH_TEST_CASE("gatedDeltaNetPrefill groups value heads", "[op][cpu][gated_delta_net]") {
  // Three value heads per key head, the shape Qwen3.5 uses.
  checkAgainstRecurrence(makeInputs({70, 33}, 2, 3, 16, true, 0x400), 2e-3f);
}

CATCH_TEST_CASE("gatedDeltaNetPrefill handles an empty sequence", "[op][cpu][gated_delta_net]") {
  checkAgainstRecurrence(makeInputs({0, 20, 0}, 1, 2, 8, true, 0x500), 1e-3f);
}

CATCH_TEST_CASE("gatedDeltaNetPrefill spans many chunks", "[op][cpu][gated_delta_net]") {
  // Four chunks and a one-token tail: the state has to be carried from one to the next four times,
  // and the last chunk is as short as a chunk gets.
  checkAgainstRecurrence(makeInputs({257}, 1, 1, 16, true, 0x600), 3e-3f);
}

CATCH_TEST_CASE("gatedDeltaNetPrefill takes the decay regimes", "[op][cpu][gated_delta_net]") {
  // No decay at all: every exp(c_i - c_j) is one, and nothing that takes a difference of logs may
  // drift off it.
  checkAgainstRecurrence(makeInputs({130}, 1, 2, 16, true, 0x700, Regime::kNoDecay), 2e-3f);

  // Decay strong enough that exp(c) underflows inside a chunk. Everything it kills is legitimately
  // zero; what must survive is the tokens whose own decay is still O(1).
  checkAgainstRecurrence(makeInputs({130}, 1, 2, 16, true, 0x710, Regime::kStrongDecay), 2e-3f);

  // Beta at both ends of what the delta rule allows: half the tokens write nothing at all, the
  // other half overshoot, which is the regime where (I + A) is furthest from the identity.
  checkAgainstRecurrence(makeInputs({130}, 1, 2, 16, true, 0x720, Regime::kBetaEnds), 5e-3f);
}

CATCH_TEST_CASE("gatedDeltaNetPrefill takes a scattered state pool", "[op][cpu][gated_delta_net]") {
  // A pool with twice the slots the batch needs and a mapping that is neither monotonic nor
  // contiguous, over a batch with empty sequences at both ends. Every slot no sequence maps to has
  // to come back exactly as it went in, which the whole-pool comparison checks.
  checkAgainstRecurrence(
      makeInputs({0, 70, 1, 0, 65, 0}, 2, 1, 16, true, 0x730, Regime::kDefault,
                 SlotPolicy::kScattered),
      2e-3f);
}

}  // namespace fl
