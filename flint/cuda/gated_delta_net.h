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

/// Which of the two implementations to run. They compute the same thing and are checked against
/// each other; neither is faster everywhere, so kAuto picks by how well the batch packs the device
/// and the forced values exist for the tests and the benchmark.
///
/// kFused gives one CTA a (sequence, value head) from that sequence's first token to its last, and
/// builds, solves and scans every chunk without going back to global memory for anything but q, k,
/// v and the output. Holding the whole (D, D) state in shared memory is what pays for that, and it
/// costs twice: only one CTA is resident per SM, so a batch with fewer (sequence, head) pairs than
/// the device has SMs leaves SMs idle for the whole prefill; and so little shared memory is left
/// over that the chunk has to be 32 rather than 64, which halves the tiles the chunk-sized phases
/// get to work with.
///
/// kChunked runs three launches -- build every chunk's system, solve the whole batch of them
/// through triangularSolveInplace, scan each sequence's chunks in order. It writes and re-reads
/// several hundred megabytes of intermediates for a long prefill, but it has blocks to spare
/// however short the batch is, and it can afford the wider chunk.
///
/// Measured on an RTX 5060 Ti over 4096 tokens of the Qwen3.5 layer (48 value heads, D=128), in
/// microseconds:
///
///   sequences     1      2      4      8
///   kChunked   7630   7266   6900   6503
///   kFused     8774   6601   6594   6049
/// kFusedRegisters is kFused with the state held in registers rather than shared memory, which is
/// what lets two CTAs be resident per SM instead of one. It pays for that with a warp shuffle
/// reduction: K S_0 and Q S_0 reduce over the axis the state is partitioned along, so the partial
/// each thread can form has to be summed across the threads that split it.
///
/// The two tensor core paths are the ones that are not a rearrangement of the same FP32 arithmetic.
/// Both are fused like kFused and put every product bigger than a vector on an HMMA, including the
/// chunk's solve, which they turn into a GEMM by inverting (I + A) explicitly rather than
/// substituting through it.
///
/// kTensorCoreMma is the faster of the two and what kAuto picks wherever it fits, which is head
/// dimensions of 32, 64 and 128. It holds the state transposed and drives mma.sync directly, so the
/// state, the right hand side, u and the output are passed from one product to the next in the
/// registers they land in: an accumulator pair, cast to half, is the A operand of the tile it
/// covers. Nothing but genuine operands goes to shared memory.
///
/// kTensorCore is the same algorithm on the WMMA API, which cannot feed an accumulator back in as
/// an operand -- so the state, the right hand side and u each make a round trip through shared
/// memory. It is ~30% slower and takes any head dimension that is a multiple of 16 up to 128.
enum class GatedDeltaNetPath {
  kAuto,
  kFused,
  kFusedRegisters,
  kChunked,
  kTensorCore,
  kTensorCoreMma,
};

/// Gated DeltaNet linear attention over a packed (varlen) batch. See F::gatedDeltaNetPrefill.
///
/// q, k, v and the returned output are <half>, g, beta and state are <float>, and cuSeqlens and
/// stateSlots are <int>. Both paths want a head dimension that is a multiple of 32; the fused one
/// also needs it to be at most 128, since the state has to fit in one CTA's shared memory.
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

}  // namespace cuda
}  // namespace op
}  // namespace fl
