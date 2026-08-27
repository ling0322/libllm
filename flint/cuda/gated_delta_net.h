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

#pragma once

#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

/// Which implementation to run. There is one, so this says what it does with the batch rather than
/// which kernel to reach for; kAuto and kTensorCoreMma are the same thing, and the third value
/// exists for the tests and the benchmark.
///
/// Four other implementations used to sit here: a WMMA version of the same tensor core algorithm
/// reaching back to sm_70, and three FP32 ones -- a three-launch chunked path with no head
/// dimension limit, a fused path holding the whole (D, D) state in shared memory, and the same
/// fused path with the state in registers. Every one of them was between four and five times
/// slower than this wherever both ran, and none of them was reachable except through a head
/// dimension or a compute capability this one does not take. They were deleted together with the
/// support for those: the operator is now head dimensions of 32, 64 and 128 on Ampere and later,
/// and anything else is turned away by the check at the top of gatedDeltaNetPrefill rather than
/// quietly routed to something a fifth as fast. The CPU reference in flint/cpu is what the tests
/// check against and always was, so nothing lost its oracle along with them.
///
/// kTensorCoreMma gives one CTA a (sequence, value head) from that sequence's first token to its
/// last, and puts every product bigger than a vector on an HMMA -- including the chunk's solve,
/// which it turns into a GEMM by inverting (I + A) explicitly rather than substituting through it.
/// It holds the state transposed and drives mma.sync directly, so the state, the right hand side, u
/// and the output are passed from one product to the next in the registers they land in: an
/// accumulator pair, cast to half, is the A operand of the tile it covers. Nothing but genuine
/// operands goes to shared memory.
///
/// `mma.sync.m16n8k16` and `ldmatrix` are what it is built out of and both start at sm_80, which is
/// where the operator's compute capability floor comes from; `tensorCoreAvailable` below is what
/// answers that half of it, and gdnmma::fits answers the whole of it.
///
/// kTensorCoreMma also carries a second path for sequences short enough to be a decode step, which
/// it steps one token at a time out of the registers the state already sits in rather than paying
/// for a chunk that is almost all padding. The branch is per CTA inside the one launch; see the
/// file comment in gated_delta_net_mma.cu for why it is not a second launch. kTensorCoreMmaChunkOnly
/// turns it off, which is what keeps the chunk path covered at the lengths that would otherwise
/// always take the other one, and is what the second path is measured against:
///
///   sequences, one token each        1      8     32    128
///   kTensorCoreMmaChunkOnly       22.9  121.8  554.0  2157
///   kTensorCoreMma                 8.1   59.3  535.3  2147
///
/// on an RTX 5060 Ti at the Qwen3.5 shape, one launch, in microseconds. The gap closes as the
/// batch grows because what is left of the launch by then is the state itself: 128 sequences of 48
/// heads is 393MB of it read and written once, which is what 2147us buys at this card's bandwidth,
/// and no arithmetic the branch removes is worth anything against that. The small columns are the
/// ones the branch is for, and they are also warm -- a batch of one is 3MB of state and the
/// benchmark launches it repeatedly, so both paths are reading L2 there, and the ratio between
/// them is the honest part of those two numbers rather than either one on its own.
///
/// The crossover, found by lengthening the sequences until stepping costs what chunking does, is
/// around 18 tokens at 8 sequences and around 13 at 64, where bandwidth is already the limit.
enum class GatedDeltaNetPath {
  kAuto,
  kTensorCoreMma,
  kTensorCoreMmaChunkOnly,
};

/// Whether this device is new enough to run this operator at all. That is a question about the
/// device and nothing else: the head dimension and the shared memory are the other half of it and
/// belong to gdnmma::fits, which gatedDeltaNetPrefill checks. This exists so a caller can ask
/// before building the tensors, and so the failure message can say which half is wrong.
bool tensorCoreAvailable();

/// Gated DeltaNet linear attention over a packed (varlen) batch. See F::gatedDeltaNetPrefill.
///
/// q, k, v and the returned output are <half>, g, beta and state are <float>, and cuSeqlens and
/// stateSlots are <int>. The head dimension has to be 32, 64 or 128 -- the kernel is a template on
/// it and those are what it is instantiated for -- and the device has to be Ampere or later.
///
/// `state` is a pool of (nVHead, D, D) slots rather than one state per sequence, and `stateSlots`
/// says which slot each sequence in this batch owns -- the same indirection a paged KV cache
/// reaches its blocks through, at a granularity of one slot per sequence, since a linear
/// attention's whole history is that fixed-size state. Nothing in the kernels depends on a
/// sequence keeping its position in the batch: every read and write of the state goes through
/// `stateSlots[seq]`, so a slot survives across the prefill chunks of one sequence and the pool
/// can be allocated and recycled by the same block manager that owns the paged cache.
Tensor gatedDeltaNetPrefill(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &g,
    const Tensor &beta,
    const Tensor &cuSeqlens,
    const Tensor &stateSlots,
    Tensor &state,
    GatedDeltaNetPath path = GatedDeltaNetPath::kAuto);

/// The kernel behind the operator above, which lives in gated_delta_net_mma.cu and is reached only
/// through it. It is a namespace of its own rather than more of this one because `fits` and `run`
/// are not names that mean anything at the scope the operator has, and it is declared here rather
/// than in a header of its own because there is one implementation left to declare.
namespace gdnmma {

/// The compute capability this path starts at. `mma.sync.m16n8k16` and `ldmatrix` are Ampere
/// instructions, and the kernel is nothing but those, so below this there is no partial version of
/// it to fall back on -- the whole body compiles away and `fits` refuses.
constexpr int kMinArch = 80;

/// The gated DeltaNet prefill on `mma.sync`, with the state and every intermediate held in
/// registers. See the file comment in gated_delta_net_mma.cu.
///
/// Returns false, and leaves `smemOut` alone, if this device cannot run it at this head dimension.
bool fits(int headDim, int *smemOut);

/// The longest sequence `run` will let take the in-kernel recurrent path rather than the chunk
/// machinery. It is a cap on the staging buffer the tokens go through -- four times this many
/// tokens' worth of k, q, v and output have to fit the buffers the chunk path stages K, Q and v in
/// -- so `recurrentMax` is clamped to it. It is a shared memory bound, not the crossover.
constexpr int kMaxRecurrentLen = 16;

/// Where the crossover is taken to be. Below it a sequence is cheaper stepped than chunked; see the
/// measurements above. The two regimes put it at 18 and at 13, and this is under both of them: what
/// is on the other side of it is a band of lengths where the step is at best a wash, and every
/// length a decode step actually has -- one token, or one plus a few speculated ones -- is well
/// inside it.
constexpr int kDefaultRecurrentLen = 12;

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
    int recurrentMax = kDefaultRecurrentLen);
}  // namespace gdnmma

}  // namespace cuda
}  // namespace op
}  // namespace fl
