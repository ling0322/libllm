//! The operations of `flint/functional.h`.
//!
//! Every function here takes its inputs by reference and returns a new tensor, except for the few
//! that write into a tensor the caller already has, which take it as `&mut`. That `&mut` is a
//! statement of intent rather than a guarantee: a tensor shares its storage with its clones and
//! its views, so an operation that writes in place can be observed through any of them.
//!
//! An operation runs on the device its inputs live on, and all of them must agree on that device.
//!
//! # Shapes are checked fatally
//!
//! These functions return a [`Result`], but it does not cover as much as a Rust API usually would.
//! The operators check the shapes and dtypes they are handed with the C++ library's own fatal
//! check, which prints a message and aborts the process; so does reaching an operation the device
//! has no kernel for. Neither is an unwind the binding could catch and turn into an `Err`, so the
//! shapes each function documents are a requirement on the caller rather than something to hand
//! over and let the error path sort out.
//!
//! ```no_run
//! use flint::{functional as F, Tensor};
//!
//! let a = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0])?;
//! let b = Tensor::from_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?;
//! assert_eq!(F::matmul(&a, &b)?.to_vec_f32()?, vec![1.0, 2.0, 3.0, 4.0]);
//! # Ok::<(), flint::Error>(())
//! ```

use crate::{check, ffi, init, DType, Device, Result, Tensor};

/// Reduce over the last dimension, the default of [`sum`] and [`max`].
pub const LAST_DIM: i32 = -1;

/// A 1-D tensor holding the values of `[begin, end)` taken `step` at a time.
pub fn arange(begin: i64, end: i64, step: i64, device: Device) -> Result<Tensor> {
    init();
    Tensor::produce(|out| unsafe { ffi::fl_arange(begin, end, step, device as i32, out) })
}

/// A tensor filled with uniform random numbers in `[0, 1)`.
pub fn rand(shape: &[i32], dtype: DType, device: Device) -> Result<Tensor> {
    init();
    Tensor::produce(|out| unsafe {
        ffi::fl_rand(
            shape.as_ptr(),
            shape.len() as i32,
            dtype as i32,
            device as i32,
            out,
        )
    })
}

/// A float tensor drawn from a normal distribution with mean 0 and variance 1.
pub fn randn(shape: &[i32], device: Device) -> Result<Tensor> {
    init();
    Tensor::produce(|out| unsafe {
        ffi::fl_randn(shape.as_ptr(), shape.len() as i32, device as i32, out)
    })
}

/// Seed the generator that [`rand`] and [`randn`] draw from on `device`.
pub fn manual_seed(device: Device, seed: u64) -> Result<()> {
    init();
    check(unsafe { ffi::fl_manual_seed(device as i32, seed) })
}

/// The rows of `table` `<float>(V, D)` named by `indices` `<long>(..)`, which gains a trailing
/// dimension of `D`.
pub fn lookup(table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_lookup(table.raw, indices.raw, out) })
}

/// Apply NeoX-style rotary embedding to `query` and `key` in place.
///
/// `positions` is `<long>(numTokens)`, `query` and `key` are `<float>(numTokens, numHeads,
/// headDim)`, and `rotary_cache` is `<float>(maxPositions, 2 * headDim)`, each row holding a
/// cosine half followed by a sine half.
pub fn rotary_embedding(
    positions: &Tensor,
    query: &mut Tensor,
    key: &mut Tensor,
    rotary_cache: &Tensor,
) -> Result<()> {
    check(unsafe { ffi::fl_rotary_embedding(positions.raw, query.raw, key.raw, rotary_cache.raw) })
}

/// Root mean square layer normalization over the last dimension of `input`, scaled by `weight`
/// `<float>(D)`.
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_rms_norm(input.raw, weight.raw, eps, out) })
}

/// Matrix multiplication, batched over the leading dimensions.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_matmul(a.raw, b.raw, out) })
}

/// Element-wise `a * b`, broadcasting `b` over the leading dimensions of `a`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_mul(a.raw, b.raw, out) })
}

/// Element-wise `a + b`, broadcasting `b` over the leading dimensions of `a`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_add(a.raw, b.raw, out) })
}

/// Element-wise `a - b`, broadcasting `b` over the leading dimensions of `a`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_sub(a.raw, b.raw, out) })
}

/// Element-wise `a == b`, as a [`DType::Bool`] tensor of the same shape.
pub fn eq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_eq(a.raw, b.raw, out) })
}

/// Element-wise `input * other`.
pub fn mul_scalar(input: &Tensor, other: f32) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_mul_scalar(input.raw, other, out) })
}

/// Element-wise `input / other`.
pub fn div_scalar(input: &Tensor, other: f32) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_div_scalar(input.raw, other, out) })
}

/// Element-wise `input % other`, for a [`DType::Long`] tensor.
pub fn mod_scalar(input: &Tensor, other: i64) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_mod_scalar(input.raw, other, out) })
}

/// Element-wise `input` squared.
pub fn square(input: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_square(input.raw, out) })
}

/// Softmax over the last dimension.
pub fn softmax(input: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_softmax(input.raw, out) })
}

/// Swish-gated linear unit over the last dimension of `input` `<float>(..., D)`, which must be
/// even and which the result halves: `swiglu(x) = swish(x[..D / 2]) * x[D / 2..]`.
pub fn swiglu(input: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_swiglu(input.raw, out) })
}

/// Sum over dimension `dim`, which may be negative to count from the back and which the result
/// drops. Pass [`LAST_DIM`] for the common case.
pub fn sum(input: &Tensor, dim: i32) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_sum(input.raw, dim, out) })
}

/// The largest element of dimension `dim`, which the result drops the same way [`sum`] does.
pub fn max(input: &Tensor, dim: i32) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_max(input.raw, dim, out) })
}

/// Concatenate `a` and `b` along `dim`. They must agree on every other dimension.
pub fn cat(a: &Tensor, b: &Tensor, dim: i32) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_cat(a.raw, b.raw, dim, out) })
}

/// A `<float>(max_len, max_len)` mask holding `-inf` where a position may not attend and `0`
/// where it may.
pub fn causal_mask(max_len: i32, device: Device) -> Result<Tensor> {
    init();
    Tensor::produce(|out| unsafe { ffi::fl_causal_mask(max_len, device as i32, out) })
}

/// Scaled dot product attention.
///
/// `q` is `<float>(N, nHead, L, D)` and `k` and `v` are `<float>(N, nKvHead, S, D)`, where
/// `nKvHead` may divide `nHead` for grouped-query attention rather than being expanded first.
/// `causal` masks the future positions, aligned to the bottom right of the score matrix.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_attention(q.raw, k.raw, v.raw, causal as i32, out) })
}

/// The keys and values of one forward pass, and where they belong in a paged KV cache.
///
/// Held together because [`paged_attention`] needs all six of them, and passing them positionally
/// makes two tensors of the same shape easy to swap by mistake.
pub struct PagedKvCache<'a> {
    /// `<float>(nBlock, blockSize, nKvHead, D)`: the key block pool.
    pub key_cache: &'a Tensor,
    /// `<float>(nBlock, blockSize, nKvHead, D)`: the value block pool.
    pub value_cache: &'a Tensor,
    /// `<int>(nSeq, maxNumBlock)`: the blocks each sequence owns, in token order.
    pub block_table: &'a Tensor,
    /// `<int>(nSeq + 1)`: the exclusive prefix sum of the query lengths.
    pub cu_seqlens_q: &'a Tensor,
    /// `<int>(nSeq)`: the number of cached tokens each sequence attends to.
    pub seqlens_k: &'a Tensor,
    /// The longest query length in the batch.
    pub max_q_len: i32,
    /// The largest value in `seqlens_k`.
    pub max_k_len: i32,
}

/// Scaled dot product attention over a packed batch of queries reading a paged KV cache.
///
/// `q` is `<float>(totalQLen, nHead, D)`, the queries of every sequence packed back to back.
/// Sequence `i` owns the blocks named by row `i` of `cache.block_table` and attends to the first
/// `cache.seqlens_k[i]` tokens they hold; the tokens it had before this call are that count minus
/// its query length, which is where `causal` starts masking.
pub fn paged_attention(q: &Tensor, cache: &PagedKvCache<'_>, causal: bool) -> Result<Tensor> {
    Tensor::produce(|out| unsafe {
        ffi::fl_paged_attention(
            q.raw,
            cache.key_cache.raw,
            cache.value_cache.raw,
            cache.block_table.raw,
            cache.cu_seqlens_q.raw,
            cache.seqlens_k.raw,
            cache.max_q_len,
            cache.max_k_len,
            causal as i32,
            out,
        )
    })
}

/// Scatter the keys and values of a forward pass into a paged KV cache, so that a later
/// [`paged_attention`] reads them back.
///
/// `k` and `v` are `<float>(numTokens, nKvHead, D)`, packed like the queries, and `slot_mapping`
/// is `<int>(numTokens)` holding `blockId * blockSize + offset` for each token.
pub fn store_kv_cache(
    k: &Tensor,
    v: &Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    check(unsafe {
        ffi::fl_store_kv_cache(
            k.raw,
            v.raw,
            key_cache.raw,
            value_cache.raw,
            slot_mapping.raw,
        )
    })
}

/// Sample one label per row of `logits` `<float>(rows, vocabSize)` with per-row parameters.
///
/// `temperatures` and `top_ps` are `<float>(rows)` and `top_ks` is `<int>(rows)`. A temperature of
/// zero selects greedily, and a `top_k` of zero or less keeps every label.
pub fn sample_with_params(
    logits: &Tensor,
    temperatures: &Tensor,
    top_ks: &Tensor,
    top_ps: &Tensor,
) -> Result<Tensor> {
    Tensor::produce(|out| unsafe {
        ffi::fl_sample_with_params(logits.raw, temperatures.raw, top_ks.raw, top_ps.raw, out)
    })
}

/// Divide the logits of the tokens in `history` `<long>(N, historyLen)` by `weight`, penalizing
/// the ones already generated. `logits` `<float>(N, vocabSize)` is written in place.
pub fn repetition_penalty(logits: &mut Tensor, history: &Tensor, weight: f32) -> Result<()> {
    check(unsafe { ffi::fl_repetition_penalty(logits.raw, history.raw, weight) })
}

/// Copy the elements of `src` into `dest`, which must have the same shape.
pub fn copy(src: &Tensor, dest: &mut Tensor) -> Result<()> {
    check(unsafe { ffi::fl_copy(src.raw, dest.raw) })
}

/// Fill every element of `tensor` with `value`, in place.
pub fn fill(tensor: &mut Tensor, value: f32) -> Result<()> {
    check(unsafe { ffi::fl_fill(tensor.raw, value) })
}

/// Whether every pair of elements is within `rtol` relative and `atol` absolute tolerance. The
/// tolerances match the C++ defaults, which are looser than a bit-for-bit comparison.
pub fn all_close(a: &Tensor, b: &Tensor) -> Result<bool> {
    all_close_with_tolerance(a, b, 1e-3, 1e-5)
}

/// [`all_close`] with the tolerances spelled out.
pub fn all_close_with_tolerance(a: &Tensor, b: &Tensor, rtol: f32, atol: f32) -> Result<bool> {
    let mut value: i32 = 0;
    check(unsafe { ffi::fl_all_close(a.raw, b.raw, rtol, atol, &mut value) })?;
    Ok(value != 0)
}

/// Whether every element of a [`DType::Bool`] tensor is true.
pub fn all(tensor: &Tensor) -> Result<bool> {
    let mut value: i32 = 0;
    check(unsafe { ffi::fl_all(tensor.raw, &mut value) })?;
    Ok(value != 0)
}

/// The single element of a one-element tensor, as an `f32`.
pub fn elem(tensor: &Tensor) -> Result<f32> {
    let mut value = 0.0f32;
    check(unsafe { ffi::fl_elem(tensor.raw, &mut value) })?;
    Ok(value)
}

/// The float type the operators of `device` work in by default.
pub fn default_float_type(device: Device) -> Result<DType> {
    init();
    let mut raw: i32 = 0;
    check(unsafe { ffi::fl_get_default_float_type(device as i32, &mut raw) })?;
    DType::from_raw(raw)
}

/// Print the tensor to stdout.
pub fn print(tensor: &Tensor) -> Result<()> {
    check(unsafe { ffi::fl_print(tensor.raw) })
}
