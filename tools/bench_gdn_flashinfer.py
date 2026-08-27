# The MIT License (MIT)
#
# Copyright (c) 2024 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Time FlashInfer's gated DeltaNet prefill on the shapes flint's own benchmark uses.

The numbers this prints line up with `./build/benchmark "[gated_delta_net]"`: the same Qwen3.5
layer (16 key heads, 48 value heads, D=128), the same 4096 tokens split over the same sequence
counts, and the same protocol -- 5 warmup calls, then 20 timed ones between two CUDA events.

`--check` first verifies that FlashInfer's kernel computes the recurrence flint's operator is
tested against,

    S_t = exp(g_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t

with `scale=1`, since a timing comparison is only worth anything if both sides do the same work.
It does, to about 1e-4 at these shapes, once two conventions are lined up:

  - the decay. flint's `g` is the log decay, at most zero; FlashInfer's is the decay itself, in
    (0, 1]. Handing FlashInfer a log decay hands it a negative number it takes the log of, and
    every output comes back NaN rather than wrong, which is at least a loud way to find out.
  - the state's layout. flint stores each head's state as (key dim, value dim), FlashInfer as
    (value dim, key dim). The two are transposes of each other, and since the state is square the
    shapes agree either way -- so getting this wrong is silent, and only shows up as a state that
    is wrong by about as much as it is large.
"""

import argparse
import math

import torch

import flashinfer

KEY_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
NUM_WARMUP = 5
NUM_ITERATIONS = 20


def make_inputs(num_tokens, num_seq, dtype, device, seed=0x1234):
    """The benchmark's inputs: unit-norm q and k, a log decay at most zero, and a zero state."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device=device, dtype=torch.float32) - 0.5

    q = torch.nn.functional.normalize(rand(num_tokens, KEY_HEADS, HEAD_DIM), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(rand(num_tokens, KEY_HEADS, HEAD_DIM), dim=-1).to(dtype)
    v = rand(num_tokens, VALUE_HEADS, HEAD_DIM).to(dtype)
    g = -(rand(num_tokens, VALUE_HEADS) + 0.5) * 0.5
    beta = rand(num_tokens, VALUE_HEADS) + 0.5

    lengths = [num_tokens // num_seq * s for s in range(num_seq + 1)]
    lengths[-1] = num_tokens
    cu_seqlens = torch.tensor(lengths, device=device, dtype=torch.int32)

    state = torch.zeros(num_seq, VALUE_HEADS, HEAD_DIM, HEAD_DIM, device=device,
                        dtype=torch.float32)

    return q, k, v, g, beta, cu_seqlens, state


def reference(q, k, v, g, beta, cu_seqlens, state):
    """The token-by-token recurrence, in float, as flint's CPU operator computes it."""
    num_tokens = q.shape[0]
    ratio = VALUE_HEADS // KEY_HEADS
    out = torch.zeros(num_tokens, VALUE_HEADS, HEAD_DIM, device=q.device, dtype=torch.float32)
    s = state.float().clone()

    q, k, v = q.float(), k.float(), v.float()
    for seq in range(cu_seqlens.numel() - 1):
        for t in range(int(cu_seqlens[seq]), int(cu_seqlens[seq + 1])):
            for h in range(VALUE_HEADS):
                kt = k[t, h // ratio]
                qt = q[t, h // ratio]
                # S <- exp(g) (I - beta k k^T) S + beta k v^T, one rank-one update of the state.
                sm = s[seq, h] * math.exp(float(g[t, h]))
                pred = kt @ sm
                sm = sm + float(beta[t, h]) * torch.outer(kt, v[t, h].float() - pred)
                s[seq, h] = sm
                out[t, h] = qt @ sm

    return out, s


def check(dtype, device):
    """Run both on a small case and report how far apart they land."""
    num_tokens, num_seq = 96, 2
    q, k, v, g, beta, cu_seqlens, state = make_inputs(num_tokens, num_seq, dtype, device)

    out_state = torch.empty_like(state)
    out = flashinfer.chunk_gated_delta_rule(
        q, k, v, g=g.exp(), beta=beta, scale=1.0, initial_state=state, output_final_state=True,
        cu_seqlens=cu_seqlens, output_state=out_state)
    if isinstance(out, tuple):
        out = out[0]

    ref_out, ref_state = reference(q, k, v, g, beta, cu_seqlens, state)
    print("output   max abs diff %.4g" % (out.float() - ref_out).abs().max().item())
    print("state    max abs diff %.4g" % (
        out_state.float() - ref_state.transpose(-1, -2)).abs().max().item())


def benchmark(num_tokens, num_seq, dtype, device):
    q, k, v, g, beta, cu_seqlens, state = make_inputs(num_tokens, num_seq, dtype, device)
    out = torch.empty(num_tokens, VALUE_HEADS, HEAD_DIM, device=device, dtype=dtype)
    out_state = torch.empty_like(state)
    # The decay, not its log: see the note at the top. Converted once, outside the timed loop,
    # since flint's kernels take the log decay and would otherwise be timed against less work.
    decay = g.exp()

    def run():
        flashinfer.chunk_gated_delta_rule(
            q, k, v, g=decay, beta=beta, scale=1.0, initial_state=state, output_final_state=True,
            cu_seqlens=cu_seqlens, output=out, output_state=out_state)

    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()

    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(NUM_ITERATIONS):
        run()
    end.record()
    end.synchronize()

    micros = begin.elapsed_time(end) * 1000.0 / NUM_ITERATIONS
    print("%-44s %10.3f us" % (
        "gated_delta_net flashinfer tokens=%d seqs=%d" % (num_tokens, num_seq), micros))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seqs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--check", action="store_true",
                        help="compare against the reference recurrence before timing")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    major, minor = torch.cuda.get_device_capability()
    print("%s, sm%d%d, flashinfer %s, %s" % (
        torch.cuda.get_device_name(), major, minor, flashinfer.__version__, args.dtype))

    if args.check:
        check(dtype, device)

    for num_seq in args.seqs:
        benchmark(args.tokens, num_seq, dtype, device)


if __name__ == "__main__":
    main()
