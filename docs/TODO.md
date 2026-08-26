# TODO

Known issues and deferred work. Each entry says what was observed, not what it might be, so
whoever picks it up starts from evidence rather than from a guess.

## `tensor_functional` failed once and did not reproduce

Seen once during `cargo test` on the branch that folded `flint-rs` into `llm` (2026-08-25).
One test in `llm/tests/tensor_functional.rs` failed:

```
test result: FAILED. 18 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
error: test failed, to rerun pass `-p llm --test tensor_functional`
```

The failing test's name and its panic output were not captured, which is the first thing to fix
if it happens again — run the suite so the failure output is kept rather than re-running to try
to reproduce it.

It did not reproduce in 8 runs of `cargo test -p llm --test tensor_functional` alone, nor in 6
runs of the full `cargo test`. So it is rare, and running that binary on its own may not be
enough to trigger it.

Unknown so far:

- whether it predates the `flint-rs` merge. Nothing in that merge changed the binding code
  itself, only the paths it is reached by, so a pre-existing flake is the more likely reading —
  but that has not been checked against `main`.
- whether it depends on other test binaries having run first. Both times the suite was run in
  full; the isolated runs that passed were of that one binary.

Worth knowing while investigating: tests inside one binary run on several threads by default,
and a flint `Tensor` is deliberately neither `Send` nor `Sync` because the operators keep
per-device state. `init()` is guarded by a `Once`. That makes ordering and startup races the
first place to look.

## Raise or remove the llama test's token cap

`llm/tests/llama.rs` skips reference cases longer than `MAX_TOKENS_PER_CASE`, which currently
means the 1271-token case never runs. That case took ~100s of the ~117s the test spent, and it
is the only one that spans more than one paged-attention block, so the cap trades away real
coverage for speed.

The cost is not the forward pass — it is comparing every position's logits, `num_tokens * 128256`
floats against two reference tensors of ~650MB each. Sampling a subset of positions instead
would keep the long-sequence coverage without the bill.

The constant's doc comment carries the same note next to the code.

## `llama.rs` says it is `#[ignore]`d but is not

The module comment in `llm/tests/llama.rs` reads:

```
//! CUDA, so this is `#[ignore]`d: run it with `cargo test --test llama -- --ignored`
```

Neither test in the file actually carries `#[ignore]`, so both run under a plain `cargo test` and
both need the model packages under `models/`, which are not in the repository. Either restore the
attribute or correct the comment — right now a checkout without those packages fails the default
test run for a reason the comment says should not apply.

## Operators the CPU backend still does not implement

Found while filling in the element-wise operators. Each of these is declared on `Operators`, is
reachable through `flint/functional.h`, and aborts with `NOT_IMPL()` when the tensor is on the CPU,
because `CPUOperators` does not override it and the base method has no fallback:

- `rotaryEmbedding`
- `pagedAttention`
- `storeKVCache`
- `matmulNarrowPrecision` (the mxfp4 path)

These are the paged-KV-cache and quantised kernels, so unlike the ones already filled in they need
real CPU implementations rather than a few lines of glue. `attention` is not on the list: it has a
working default on `Operators` built out of matmul and softmax, so the CPU gets it for free.

Nothing in the Rust runtime hits these on CPU today — the model runs on CUDA — so this only matters
for running a model on CPU, or for testing a CUDA kernel against a CPU reference the way the
element-wise tests now do.

## Operators worth adding next

The element-wise set, `div(a, b)` and `min` are in. Still missing from `functional.h`, roughly in
order of how often an inference engine wants them:

- `argmax`. Greedy decoding goes through `sample` with a temperature of 0 today, which works but
  makes the caller build three parameter tensors to ask for the largest logit. Unlike the
  reductions already there it returns indices rather than values, so it needs its own kernel that
  carries an index through the block reduction rather than an extra `MapReduceType`.
- `layerNorm`. `rmsNorm` covers Llama-style models; a model family that normalises with a mean and
  a variance cannot run at all.
- comparisons other than `eq` (`gt`, `lt`, `ge`, `le`, `ne`). `eq` is also unusual in taking only
  `<uint8>` and answering in `<bool>`, which is narrower than it looks.
- `mean`, `clamp`, `pow`, `topk`, `where`/`masked_fill`.

## `eq` only compares uint8 tensors

`Operators::eq` is implemented for `<uint8>` on both backends and nothing else, which is narrow
enough to be surprising given the name — comparing two `<float>` tensors is the obvious use and it
aborts. Widening it means picking a rounding policy for float comparison, which is why it was left
as-is rather than extended along with the other element-wise work.

## Gated DeltaNet prefill is within 10% of FlashInfer

`F::gatedDeltaNetPrefill` has five CUDA implementations. `kAuto` picks `gdnmma`
(`flint/cuda/gated_delta_net_mma.cu`) wherever it fits, which is head dimensions of 32, 64 and 128
on Ampere or later. Behind it are `gdnwmma`, the same algorithm on the WMMA API for the head
dimensions the first cannot take, and three FP32 SIMT paths kept for what they measured.

Measured on an RTX 5060 Ti (36 SMs, ~24 TFLOP/s FP32, ~88 TFLOP/s FP16 tensor, ~448 GB/s) with
`./build/benchmark "[gated_delta_net]"` and `tools/bench_gdn_flashinfer.py`, one Qwen3.5 gated
DeltaNet layer over 4096 tokens (16 key heads, 48 value heads, D=128), in microseconds:

```
  sequences        1      2      4      8
  chunked       7258   6868   6530   6124
  fused         8838   6650   6649   6100
  fused-regs    8530   7786   6444   6148
  tensorcore    1751   1336   1332   1245     (WMMA)
  mma           1301   1007   1031    936
  flashinfer    1094    851    897    876
```

So 6.5x faster than the best FP32 path and within 7% of FlashInfer's own SM120 kernel at eight
sequences, 19% at one. At 14 GFMA per layer that is ~30 TFLOP/s against FlashInfer's ~32, on a card
whose FP32 peak is 24 -- which is the whole point, since none of the FP32 paths could have got here
by any amount of tuning.

### The two things that closed the gap

Both are about where a product's result lives, not about the arithmetic, which has been the same
since the first tensor core version.

- **`mma.sync` instead of WMMA, with the state transposed.** WMMA will not say where an accumulator
  element is, so anything a product computes has to be written to shared memory before another
  product can read it as an operand: the state once a chunk, the right hand side once, u once. With
  `mma.sync` two adjacent m16n8 accumulators, cast to half, *are* the A operand of the 16 by 16 tile
  they cover -- so each product feeds the next out of the registers it landed in. Holding the state
  as its transpose (value dimension by key dimension, which is also how FlashInfer holds it) is what
  puts every intermediate on the A side of the products that consume it. It also puts the token on
  the *column* of every accumulator, which turns the two decays the WMMA path applies as in-place
  passes over shared memory into a multiply on a register.
- **The head dimension as a template parameter.** This is worth as much as everything above it: the
  same kernel with a runtime head dimension is **1681 us**, with 256 bytes a thread of stack frame,
  because every accumulator array is indexed by a loop bound derived from it and ptxas will not keep
  a dynamically indexed array in registers. Instantiated per head dimension it is 956 us with no
  stack at all. A register-resident design is only register-resident if the compiler can prove the
  indices.

Prefetching the next chunk's K, Q and v -- a second key buffer, and copies issued as each of the
other two falls out of use -- took the last 2%.

### What it cost to get there, and what to check first if it breaks

The layout facts below are not documented guesses; each was verified against a reference GEMM on
this device (`ldmatrix` and `mma.sync` are ABI, but which register holds what is easy to get wrong
and silent when you do -- the shapes are identical and only the answer changes):

- `ldmatrix.x4` on a row-major tile, with lane l addressing row l % 16 and column half l / 16,
  gives the four registers as (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15),
  (rows 8-15, k 8-15). As an A operand that is exactly right in register order.
- As **B operands** it is not: the operand for the first eight columns is registers 0 and 2, and for
  the second eight registers 1 and 3. Pairing them 0-1 and 2-3 is the mistake that cost the longest
  debugging round here, and it is why `mmaB` exists and is the only place the pairing is written.
- With `.trans` it is the other way round -- the transpose makes the halves of the address pattern
  index n rather than k -- so there the operands *are* consecutive. `mmaBTrans` is the only place
  that one is written.
- A 16 by 16 accumulator pair cast to half is the A operand of that tile, in register order.

The debugging that found the pairing bug is worth repeating if this ever breaks: dump the staged
tensors and compare against the input (they were exact), compute one output tile with a scalar loop
out of the same shared memory (exact), then compare the mma result against that (wrong) -- which
localises it to the instruction's operands in three steps rather than by reading PTX.

### Measurements from the WMMA path, which still apply

The WMMA path is 1245 us and the same shape of kernel, so its ablations are the best picture of
where the time goes: the HMMAs themselves ~600 us, the barriers ~181, the state's trip to shared
memory ~59, the key and query loads ~53, the two in-place decay passes ~48, the output store ~33.
On the mma path the same ablations give: global loads 93 us, the output store 46, the inverse 21.

The kernel is latency bound, not issue bound: at eight warps it issued about a sixth of the
instruction slots it had, because one CTA is resident per SM at this footprint and the block's own
warps are all there is to hide anything behind. That is why the WMMA path runs sixteen warps -- the
largest count its tiles divide into evenly -- and why the mma path, which is pinned to eight by the
row partitioning, needed the register residency instead.

What paid on the WMMA path, in order: vectorising the epilogues (3428 to 1550 us), sixteen warps
(1306 to 1244), fusing A and the score matrix into one pass, skipping the six tiles above the
diagonal, probing the accumulator layout at launch rather than round-tripping through shared memory
(1377 to 1335), the inversion moving to four warps behind a named barrier (1336 to 1306), staging v
coalesced, `cp.async` for the keys and queries, and a barrier-free decay scan.

What did not pay anywhere: twelve warps (5% slower than eight -- 64 state tiles over 12 is six for
some and five for others); giving each warp consecutive tiles to share an operand (25% fewer
fragment loads, 9% slower, because the triangular phases then hand one warp the cheapest tiles and
another the dearest); hoisting the operand the fixed column block indexes out of the tile loop (also
25% fewer loads, 12% slower, since it takes the loads out of the shadow of each other's HMMAs);
balancing the triangular phases (no change, which is what says the barrier cost is exposed load
latency rather than warps waiting on each other); staging the mma path's output through shared
memory to coalesce it; and padding the chunk-square matrices by a whole tile.

One correctness note, since it is the kind of thing tile handouts invite: a warp's tiles all sit in
one column block only when the tile width divides the warp count, which is true at D of 32, 64 and
128 and false at 48, 80 and 112. Hoisting that column out of the tile loop as a per-warp constant is
wrong at those head dimensions, silently, and there is a test at 48 for it.

### What is left

- **The last 7%, and the 19% at one sequence.** Both kernels lose the same way at one sequence: 48
  CTAs against 36 SMs is 1.33 waves, a third of the machine idle in the second one. FlashInfer is
  1094 against 876 for the same reason. Splitting a long sequence into segments and passing the
  state between them, which is what the chunked path does across launches, would fill it.
- **Warp specialisation.** FlashInfer runs 384 threads as three warp groups, one of them doing
  nothing but loads. The mma path is pinned to eight math warps by the row partitioning, so adding a
  load group is the natural next step and there is shared memory for it.
- **More head dimensions.** The mma path is instantiated for 32, 64 and 128; anything else falls to
  the WMMA path, which is 30% slower. Instantiating the rest costs compile time in proportion.

### The three FP32 paths, and what they measured

They are still reachable by name, and what they established, before any of them was the answer:

- `kChunked` is three launches -- build every chunk's system, solve the batch of them through
  `triangularSolveInplace`, scan each sequence's chunks in order. It moves several hundred megabytes
  of intermediates for a long prefill but has blocks to spare however short the batch is. Its
  scratch is proportional to the token count -- 4096 tokens at these head counts is about 150 MB --
  so a long prefill through it has to arrive in batches. Neither tensor core path has any scratch.
- `kFused` gives one CTA a (sequence, value head) and keeps the state in shared memory. It removes
  ~80% of that traffic and 22% of the multiply-adds, and it is not faster: the (D, D) state is 64 KB
  of the 99 KB a block may have, so one CTA is resident per SM, and the chunk had to shrink to 32.
- `kFusedRegisters` is the same kernel with the state in registers, so two CTAs are resident. It
  pays for that with a warp shuffle reduction and 56 bytes a thread of spill, and it is a wash.

Three arrangements of the same FP32 arithmetic landing within 15% of each other is what argued for
changing the arithmetic rather than arranging it again. Two other results from that period: padding
the staged tiles to an odd row stride took `buildChunkKernel` from 2950 to 1642 us, and carrying the
right hand sides in half rather than float took the solve from 1190 to 835.

### Lining up with FlashInfer, if you compare again

`tools/bench_gdn_flashinfer.py` times `flashinfer.chunk_gated_delta_rule` on exactly the shapes the
benchmark above uses, with the same 5 warmups and 20 timed iterations between CUDA events, and
`--check` first verifies it computes the same recurrence flint's CPU operator is tested against. It
does, to about 1e-4, once two conventions are lined up:

- FlashInfer's `g` is the decay, in (0, 1]; flint's is its log, at most zero. Passing a log decay
  gives all-NaN output, since the kernel takes a log of it.
- FlashInfer stores a head's state as (value dim, key dim), flint as (key dim, value dim). The state
  is square, so the shapes agree either way and getting it wrong is silent. The mma path holds the
  transpose in registers for exactly the reason FlashInfer does, and transposes it on the way to and
  from the state tensor, which keeps flint's own layout unchanged.

FlashInfer has the same state-pool indirection this operator has -- `state_indices` alongside a
pool-shaped `initial_state` -- but only on its SM100/SM103 kernel: on sm120 it raises
`NotImplementedError`. The comparison above is against its packed, sequence-ordered path.

## `CudaOperators::zeros` ignores the dtype it is asked for

`flint/cuda/cuda_operators.cc` builds the tensor with `createCudaTensorHalf` whatever `dtype`
says, so `zeros(shape, DType::kFloat)` hands back a `<half>` and the next operator to look at it
aborts on a dtype check. Found while writing the gated DeltaNet benchmark, which now builds its
FP32 state on the host and copies it over instead. `op::cuda::fill` is half-only, which is
presumably why it was written this way, so fixing it means giving `fill` the other types first.
