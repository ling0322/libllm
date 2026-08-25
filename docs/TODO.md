# TODO

Known issues and deferred work. Each entry says what was observed, not what it might be, so
whoever picks it up starts from evidence rather than from a guess.

## `draws_reproducible_random_numbers` races against the sampling test

`llm/tests/tensor_functional.rs::draws_reproducible_random_numbers` fails intermittently, roughly
one run in ten. Diagnosed 2026-08-25:

```
assertion `left == right` failed
  left:  [0.0009440733, 0.57136285, 0.2568094, ...]
  right: [0.57136285, 0.2568094, 0.44767365, ...]
```

The second draw is the first shifted by exactly one element, so one extra number came out of the
generator between the two seeded draws.

The cause is that `CPUOperators::sample` draws from the same generator as `rand`
(`flint/cpu/cpu_operators.cc`, `float draw = _rand.nextFloat() * selectedWeight;`) — one value per
row. The test carries a comment saying it has to stay the only test in the file that draws from the
generator, and it is the only one that draws *explicitly*; the sampling test in the same file draws
implicitly through `F::sample_with_params`. libtest runs the tests of one binary on several threads,
so that single `nextFloat()` lands between the seed and the draw.

Worth deciding rather than patching blind, which is why it is here and not fixed:

- give sampling its own generator, which decouples the two but means `manual_seed` no longer makes
  sampling reproducible — probably not what a caller expects;
- keep one generator and make it thread-safe, which serialises every draw;
- or leave the design and stop the tests from overlapping, which fixes the symptom only in this
  file and leaves the next caller to trip over the same sharing.

The underlying point is that the per-device generator is shared mutable state with no lock, so any
two threads drawing at once interfere. That is a property of the engine, not of the test.

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

## Qwen3.5 (Qwen3.8-27B), text only

Checked against the published config: 64 layers, hidden 5120, 24 query heads over 4 KV heads,
head_dim 256, intermediate 17408, silu, RMS eps 1e-6, vocab 248320, RoPE theta 1e7.

Already covered: SwiGLU, RMSNorm, grouped-query attention at 24:4, head_dim 256 (the attention
tests cover it), and the element-wise gate primitives silu/sigmoid/exp/rsqrt. Partial RoPE and the
causal convolution are now in.

What is still missing, and it is the bulk of the model:

- **Gated DeltaNet linear attention.** `full_attention_interval` is 4, so only 16 of the 64 layers
  run the attention this engine already has; the other 48 run a linear-attention layer with a
  recurrent matrix-valued state. That is a whole attention mechanism, comparable in size to the
  paged-attention work already in the tree, not an operator: a chunked scan over time carrying an
  N-by-N state, with the numerical care that goes with it. The published parameters are 16 key
  heads and 48 value heads, both with a head dimension of 128, and a convolution kernel of 4.
- **A convolution state for decode.** `F::causalConv1d` takes a whole packed batch and zero-pads at
  each sequence start, which is what prefill needs. Generating one token at a time needs the last
  `kernelSize - 1` positions of each sequence carried between steps, the way the KV cache carries
  keys and values.

Not operator work, listed so it is not forgotten: the vision tower (the checkpoint is a
`Qwen3_5ForConditionalGeneration` and carries a `vision_config`), and the built-in MTP draft head.

Two things the model needs that turned out to compose from what already exists, so they need no
kernel of their own: a gated RMSNorm is `mul(rmsNorm(x, w, eps), silu(gate))`, and an L2
normalisation over the last dimension is `mul(x, rsqrt(sum(square(x))))`. A fused kernel would be
faster for both, but neither blocks bringing the model up.
