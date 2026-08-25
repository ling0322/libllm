//! Raw declarations for the `flint/capi.h` C interface.
//!
//! Nothing here is safe to call directly: handles are raw pointers with manual lifetimes and every
//! function reports failure through a thread-local error rather than the type system. The wrappers
//! in [`crate`] exist to put those rules back.

use std::os::raw::{c_char, c_int, c_void};

pub const FL_OK: i32 = 0;
pub const FL_ERROR_INVALID_ARG: i32 = 0x0100;
pub const FL_ERROR_ABORTED: i32 = 0x0102;

/// Open end of a slice range, matching `FL_NONE`.
pub const FL_NONE: i32 = i32::MIN;

/// Opaque tensor handle. Only ever held behind a pointer.
#[repr(C)]
pub struct FlTensorImpl {
    _private: [u8; 0],
}

pub type FlTensor = *mut FlTensorImpl;

pub type FlDType = c_int;
pub type FlDeviceType = c_int;

extern "C" {
    pub fn fl_init();
    pub fn fl_is_device_available(device: FlDeviceType, out: *mut i32) -> i32;
    pub fn fl_get_last_error_code() -> i32;
    pub fn fl_get_last_error_message() -> *const c_char;

    pub fn fl_tensor_zeros(
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        device: FlDeviceType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_empty(
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        device: FlDeviceType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_from_data(
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        data: *const c_void,
        data_size: i64,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_clone(tensor: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_destroy(tensor: FlTensor);

    pub fn fl_tensor_get_dim(tensor: FlTensor, out: *mut i32) -> i32;
    pub fn fl_tensor_get_shape(tensor: FlTensor, dim: i32, out: *mut i32) -> i32;
    pub fn fl_tensor_get_stride(tensor: FlTensor, dim: i32, out: *mut i32) -> i32;
    pub fn fl_tensor_get_numel(tensor: FlTensor, out: *mut i64) -> i32;
    pub fn fl_tensor_get_dtype(tensor: FlTensor, out: *mut FlDType) -> i32;
    pub fn fl_tensor_get_device(tensor: FlTensor, out: *mut FlDeviceType) -> i32;
    pub fn fl_tensor_is_contiguous(tensor: FlTensor, out: *mut i32) -> i32;

    pub fn fl_tensor_view(
        tensor: FlTensor,
        shape: *const i32,
        ndim: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_transpose(tensor: FlTensor, dim0: i32, dim1: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_slice(
        tensor: FlTensor,
        dim: i32,
        begin: i32,
        end: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_subtensor(tensor: FlTensor, index: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_unsqueeze(tensor: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_squeeze(tensor: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_contiguous(tensor: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_to_device(tensor: FlTensor, device: FlDeviceType, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_cast(tensor: FlTensor, dtype: FlDType, out: *mut FlTensor) -> i32;

    pub fn fl_tensor_get_nbytes(tensor: FlTensor, out: *mut i64) -> i32;
    pub fn fl_tensor_copy_to_host(tensor: FlTensor, buffer: *mut c_void, buffer_size: i64) -> i32;

    pub fn fl_arange(
        begin: i64,
        end: i64,
        step: i64,
        device: FlDeviceType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_rand(
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        device: FlDeviceType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_randn(shape: *const i32, ndim: i32, device: FlDeviceType, out: *mut FlTensor) -> i32;
    pub fn fl_manual_seed(device: FlDeviceType, seed: u64) -> i32;

    pub fn fl_lookup(table: FlTensor, indices: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_rotary_embedding(
        positions: FlTensor,
        query: FlTensor,
        key: FlTensor,
        rotary_cache: FlTensor,
    ) -> i32;
    pub fn fl_rms_norm(input: FlTensor, weight: FlTensor, eps: f32, out: *mut FlTensor) -> i32;
    pub fn fl_matmul(a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_mul(a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_div(a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_add(a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sub(a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_eq(a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_mul_scalar(input: FlTensor, other: f32, out: *mut FlTensor) -> i32;
    pub fn fl_div_scalar(input: FlTensor, other: f32, out: *mut FlTensor) -> i32;
    pub fn fl_mod_scalar(input: FlTensor, other: i64, out: *mut FlTensor) -> i32;
    pub fn fl_square(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_neg(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_abs(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_exp(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sqrt(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_rsqrt(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sigmoid(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_tanh(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_relu(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_gelu(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_silu(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_softmax(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_swiglu(input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sum(input: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_max(input: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_min(input: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_cat(a: FlTensor, b: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_causal_mask(max_len: i32, device: FlDeviceType, out: *mut FlTensor) -> i32;

    pub fn fl_attention(
        q: FlTensor,
        k: FlTensor,
        v: FlTensor,
        causal: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_paged_attention(
        q: FlTensor,
        key_cache: FlTensor,
        value_cache: FlTensor,
        block_table: FlTensor,
        cu_seqlens_q: FlTensor,
        seqlens_k: FlTensor,
        max_q_len: i32,
        max_k_len: i32,
        causal: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_store_kv_cache(
        k: FlTensor,
        v: FlTensor,
        key_cache: FlTensor,
        value_cache: FlTensor,
        slot_mapping: FlTensor,
    ) -> i32;

    pub fn fl_sample_with_params(
        logits: FlTensor,
        temperatures: FlTensor,
        top_ks: FlTensor,
        top_ps: FlTensor,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_repetition_penalty(logits: FlTensor, history: FlTensor, weight: f32) -> i32;

    pub fn fl_copy(src: FlTensor, dest: FlTensor) -> i32;
    pub fn fl_fill(tensor: FlTensor, value: f32) -> i32;
    pub fn fl_all_close(a: FlTensor, b: FlTensor, rtol: f32, atol: f32, out: *mut i32) -> i32;
    pub fn fl_all(tensor: FlTensor, out: *mut i32) -> i32;
    pub fn fl_elem(tensor: FlTensor, out: *mut f32) -> i32;
    pub fn fl_get_default_float_type(device: FlDeviceType, out: *mut FlDType) -> i32;
    pub fn fl_print(tensor: FlTensor) -> i32;

    pub fn fl_memory_capture(device: FlDeviceType, out: *mut FlMemorySnapshot) -> i32;
    pub fn fl_memory_reset_peak_stats(device: FlDeviceType) -> i32;
}

/// Mirrors `fl_memory_snapshot_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FlMemorySnapshot {
    pub total: i64,
    pub free: i64,
    pub allocated: i64,
    pub peak_allocated: i64,
}
