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
