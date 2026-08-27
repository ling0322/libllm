# TODO

Known issues and deferred work. Each entry says what was observed, not what it might be, so
whoever picks it up starts from evidence rather than from a guess.

## The mma path's recurrent branch has a reproducible hump at 13 to 15 tokens

Sweeping the sequence length at 64 sequences, 48 value heads and D=128 on an RTX 5060 Ti, the
in-kernel recurrent path costs, in microseconds:

```
len   11     12     13     14     15     16
    1106   1115   1172   1205   1175   1145
```

which is not monotonic: 14 tokens costs 5% more than 16 does. It reproduced identically across
four runs, so it is not noise, and the chunk path measured over the same lengths does not have it
(1130 at 12, 1140 at 14, 1146 at 16 -- flat and rising slowly, as it should be).

Ruled out: loop unrolling of the per-token loop. Building the same sweep with `#pragma unroll 1`
on it moved nothing (1193 at 14 against 1205), so it is not the compiler choosing a different
schedule at those trip counts.

Not yet looked at: whether it survives at other head dimensions or sequence counts, and whether it
is visible in the memory counters. The case is bandwidth bound at 64 sequences -- the state alone
is 196MB read and written -- so a 5% hump is somewhere in the memory system rather than in the
arithmetic, which is what makes it worth a profile rather than a re-read of the kernel.

`kDefaultRecurrentLen` is 12 partly because of this: the band is on the other side of it. Raising
it is worth ~8% at small batch, so this is what stands in the way.

## The mma kernel costs ten more registers at a head dimension of 32

Adding the recurrent branch took the D=32 instantiation from 128 registers to 138, with no spill
either way. 128 by 256 threads is exactly half a Blackwell SM's register file, so the baseline fit
two CTAs per SM there and this one fits one. D=64 went the other way (162 to 156) and D=128, which
is the shape the model runs, did not move at all.

`__launch_bounds__(kThreads, 2)` on the D=32 instantiation does bring it back to 128, but it buys
that with 36 bytes of spill stores and 92 of spill loads, which is the worse trade -- so it is not
applied. Nothing measures D=32: the benchmarks are all at the Qwen3.5 head dimension, and it is
reached only by the tests. Whether the lost CTA costs anything there is unmeasured.

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
wrong at those head dimensions, silently. The test that covered it was at 48 and went when the WMMA
path did -- the operator no longer takes a head dimension that is not 32, 64 or 128 -- so anyone
instantiating the mma path for one of them has that trap waiting and no test standing on it.

### What is left

- **The last 7%, and the 19% at one sequence.** Both kernels lose the same way at one sequence: 48
  CTAs against 36 SMs is 1.33 waves, a third of the machine idle in the second one. FlashInfer is
  1094 against 876 for the same reason. Splitting a long sequence into segments and passing the
  state between them, which is what the chunked path does across launches, would fill it.
- **Warp specialisation.** FlashInfer runs 384 threads as three warp groups, one of them doing
  nothing but loads. The mma path is pinned to eight math warps by the row partitioning, so adding a
  load group is the natural next step and there is shared memory for it.
- **More head dimensions.** The mma path is instantiated for 32, 64 and 128 and nothing else is
  left to catch the rest -- `gatedDeltaNetPrefill` refuses them now rather than routing them
  somewhere slower. Instantiating more costs compile time in proportion; read the correctness note
  above first, since 48, 80 and 112 are exactly the ones the tile handout is a trap at.

### The three FP32 paths, and what they measured

All three are deleted, along with the WMMA path and `cuda/triangular_solve.cu`, which nothing but
`kChunked` used. This is what they established, before any of them was the answer -- kept because
it is the argument for the shape the surviving kernel has, not because the code is coming back:

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

## The hybrid cache has no model to serve yet

`llm/src/kv_cache.rs` describes a model's layers one by one, so a model whose layers do not all
attend over every token gets a pool of blocks for the layers that attend and a pool of per-request
state slots for the ones that recur. `ModelCacheSpec::qwen3_5` builds the spec of the architecture
the Qwen 3.8 models use, from the fields of its `config.json` under the names that file uses.

For the 27B (`Qwen/Qwen3.8-27B`, `model_type: qwen3_5`), the numbers are worth having written down,
because one of them decides a configuration knob:

```
64 layers, layer_types = 3 linear_attention then 1 full_attention, repeated
  16 full attention   4 key-value heads of 256          -> 64 KB of keys and values a token
  48 gated DeltaNet   48 value heads of 128 by 128      -> 3.0 MB of recurrence a layer
                      conv over 2*16*128 + 48*128 = 10240 channels, kernel 4
                                                        -> 60 KB of window a layer
max_position_embeddings 262144
```

A state slot is 147 MB — held from the moment a request is admitted to the moment it finishes,
whether it is at its first token or its millionth. That is what about 2350 tokens of context cost
in the same model, so `max_num_seqs` is no longer only a batching knob: at the default of 256 the
state pool alone would want 37 GB. Sizing it is the first thing to get right on this model, and it
is why the pools are counted separately here rather than padded to a common page size the way vLLM
does it.

What is missing is a model that asks for any of this:

the runtime implements Llama and nothing else, so nothing calls `ModelCacheSpec::qwen3_5` outside
the tests. The scheduler takes and gives back both kinds of allocation already.

To serve one, in the order it would have to be done:

- **The model.** A `qwen3.rs` beside `llama.rs`, reading its layer pattern out of the package's
  configuration rather than assuming the interval. `ModelCacheSpec::interleaved` covers "every
  fourth layer attends"; anything else hands `ModelCacheSpec::new` the layers directly.
- **The decode step of the recurrence.** `F::gatedDeltaNetPrefill` folds a whole chunk of tokens
  into the state at once, which is what a prefill wants. Decoding one token at a time against a
  slot is the same recurrence with a chunk of one, and it is not written.
- **The multi-token-prediction head.** `mtp_num_hidden_layers` is 1, and if speculative decoding is
  ever run that layer attends too, so it needs a spec of its own rather than being left out of the
  layer list.
- **Prefix caching, if it is ever added, has to skip the recurrent group.** Blocks hold what was
  computed from a prefix and two requests with the same prefix could share them; a state is the
  whole history folded together and cannot be cut at an arbitrary token. vLLM disables it per
  group for the same reason.
