//! Tests for the operations the CPU backend has no kernel for, which are the ones the tests in
//! `functional.rs` cannot reach.
//!
//! They need a CUDA device and a build configured with `WITH_CUDA`, so they are `#[ignore]`d and
//! run with `cargo test --test cuda -- --ignored`. On a machine without one, the library ends the
//! process rather than reporting an error, which is why they do not simply detect and skip.

use flint::{functional as F, DType, Device, Tensor};

/// The float type the CUDA operators work in, which the inputs have to be in already.
fn cuda_float() -> DType {
    F::default_float_type(Device::Cuda).unwrap()
}

/// Moves row-major float data onto the device, in the float type the operators there expect.
fn cuda_f32(shape: &[i32], data: &[f32]) -> Tensor {
    Tensor::from_f32(shape, data)
        .unwrap()
        .cast(cuda_float())
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap()
}

fn to_host_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap()
}

#[test]
#[ignore = "needs a CUDA device"]
fn counts_with_arange() {
    let x = F::arange(0, 6, 2, Device::Cuda).unwrap();

    assert_eq!(x.dtype(), DType::Long);
    assert_eq!(
        x.to_device(Device::Cpu).unwrap().to_vec_i64().unwrap(),
        vec![0, 2, 4]
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn squares_and_divides() {
    let x = cuda_f32(&[3], &[1.0, 2.0, 3.0]);

    assert_eq!(to_host_f32(&F::square(&x).unwrap()), vec![1.0, 4.0, 9.0]);
    assert_eq!(
        to_host_f32(&F::div_scalar(&x, 2.0).unwrap()),
        vec![0.5, 1.0, 1.5]
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn takes_a_remainder() {
    let x = F::arange(0, 5, 1, Device::Cuda).unwrap();
    let remainders = F::mod_scalar(&x, 2).unwrap();

    assert_eq!(
        remainders
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec_i64()
            .unwrap(),
        vec![0, 1, 0, 1, 0]
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn compares_element_wise() {
    // The comparison kernel works on bytes rather than on the float type of the device.
    let on_cuda = |data: &[u8]| {
        Tensor::from_u8(&[4], data)
            .unwrap()
            .to_device(Device::Cuda)
            .unwrap()
    };
    let a = on_cuda(&[1, 2, 3, 4]);
    let b = on_cuda(&[1, 0, 3, 0]);

    assert!(F::all(&F::eq(&a, &a).unwrap()).unwrap());

    let mask = F::eq(&a, &b).unwrap();
    assert_eq!(mask.dtype(), DType::Bool);
    assert!(!F::all(&mask).unwrap());
    assert_eq!(
        mask.to_device(Device::Cpu).unwrap().to_vec_bool().unwrap(),
        vec![true, false, true, false]
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn reads_a_single_element() {
    let x = cuda_f32(&[1], &[2.5]);

    assert_eq!(F::elem(&x).unwrap(), 2.5);
}

#[test]
#[ignore = "needs a CUDA device"]
fn draws_normal_numbers() {
    // The generator of a device is shared by everything running on it, and the other tests here
    // draw from it too, so reseeding cannot be checked from this binary. The CPU tests cover it.
    F::manual_seed(Device::Cuda, 7).unwrap();
    let x = F::randn(&[256], Device::Cuda).unwrap();

    assert_eq!(x.shape(), vec![256]);
    let values = to_host_f32(&x);
    assert!(
        values.iter().all(|x| x.is_finite()),
        "unexpected draw: {values:?}"
    );

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    assert!(
        mean.abs() < 0.5,
        "a standard normal should centre on zero, not {mean}"
    );
    assert!(
        values.iter().any(|x| x.abs() > 1.0),
        "a standard normal should spread out: {values:?}"
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn samples_from_logits() {
    let logits = cuda_f32(&[2, 3], &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let temperatures = Tensor::from_f32(&[2], &[0.0, 0.0])
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();
    let top_ks = Tensor::from_i32(&[2], &[0, 0])
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();
    let top_ps = Tensor::from_f32(&[2], &[1.0, 1.0])
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();

    let labels = F::sample_with_params(&logits, &temperatures, &top_ks, &top_ps).unwrap();
    assert_eq!(
        labels.to_device(Device::Cpu).unwrap().to_vec_i64().unwrap(),
        vec![1, 2]
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn rotates_query_and_key_in_place() {
    const NUM_TOKENS: i32 = 3;
    const NUM_HEADS: i32 = 2;
    const HEAD_DIM: i32 = 64;

    let positions = Tensor::from_i64(&[NUM_TOKENS], &[0, 1, 2])
        .unwrap()
        .to_device(Device::Cuda)
        .unwrap();
    let mut query = F::rand(
        &[NUM_TOKENS, NUM_HEADS, HEAD_DIM],
        cuda_float(),
        Device::Cuda,
    )
    .unwrap();
    let mut key = F::rand(
        &[NUM_TOKENS, NUM_HEADS, HEAD_DIM],
        cuda_float(),
        Device::Cuda,
    )
    .unwrap();
    let cache = F::rand(&[8, 2 * HEAD_DIM], cuda_float(), Device::Cuda).unwrap();

    let before = to_host_f32(&query);
    F::rotary_embedding(&positions, &mut query, &mut key, &cache).unwrap();
    let after = to_host_f32(&query);

    assert_eq!(query.shape(), vec![NUM_TOKENS, NUM_HEADS, HEAD_DIM]);
    // Position 0 rotates by nothing, so the first token is the one that has to stay put.
    let row = (NUM_HEADS * HEAD_DIM) as usize;
    assert_ne!(
        before[row..],
        after[row..],
        "later tokens should have moved"
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn attends_over_a_batch() {
    const HEAD_DIM: i32 = 64;

    // One value per key position, so attention picks out a mixture of known rows.
    let q = F::rand(&[1, 1, 2, HEAD_DIM], cuda_float(), Device::Cuda).unwrap();
    let k = F::rand(&[1, 1, 2, HEAD_DIM], cuda_float(), Device::Cuda).unwrap();
    let v = F::rand(&[1, 1, 2, HEAD_DIM], cuda_float(), Device::Cuda).unwrap();

    let out = F::attention(&q, &k, &v, true).unwrap();
    assert_eq!(out.shape(), vec![1, 1, 2, HEAD_DIM]);

    // With a causal mask the first query sees only the first value, unchanged.
    let first_out = &to_host_f32(&out)[..HEAD_DIM as usize];
    let first_v = &to_host_f32(&v)[..HEAD_DIM as usize];
    assert!(
        first_out
            .iter()
            .zip(first_v)
            .all(|(a, b)| (a - b).abs() < 5e-3),
        "the first row should be the first value"
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn stores_and_reads_a_paged_kv_cache() {
    const HEAD_DIM: i32 = 64;
    const BLOCK_SIZE: i32 = 16;
    const NUM_BLOCKS: i32 = 2;
    const NUM_TOKENS: i32 = 4;

    let q = F::rand(&[NUM_TOKENS, 1, HEAD_DIM], cuda_float(), Device::Cuda).unwrap();
    let k = F::rand(&[NUM_TOKENS, 1, HEAD_DIM], cuda_float(), Device::Cuda).unwrap();
    let v = F::rand(&[NUM_TOKENS, 1, HEAD_DIM], cuda_float(), Device::Cuda).unwrap();

    let mut key_cache = F::rand(
        &[NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
        cuda_float(),
        Device::Cuda,
    )
    .unwrap();
    let mut value_cache = F::rand(
        &[NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
        cuda_float(),
        Device::Cuda,
    )
    .unwrap();

    let on_cuda = |shape: &[i32], data: &[i32]| {
        Tensor::from_i32(shape, data)
            .unwrap()
            .to_device(Device::Cuda)
            .unwrap()
    };
    // The whole sequence lands at the front of block 0.
    let slot_mapping = on_cuda(&[NUM_TOKENS], &[0, 1, 2, 3]);
    F::store_kv_cache(&k, &v, &mut key_cache, &mut value_cache, &slot_mapping).unwrap();

    let block_table = on_cuda(&[1, NUM_BLOCKS], &[0, 1]);
    let cu_seqlens_q = on_cuda(&[2], &[0, NUM_TOKENS]);
    let seqlens_k = on_cuda(&[1], &[NUM_TOKENS]);
    let cache = F::PagedKvCache {
        key_cache: &key_cache,
        value_cache: &value_cache,
        block_table: &block_table,
        cu_seqlens_q: &cu_seqlens_q,
        seqlens_k: &seqlens_k,
        max_q_len: NUM_TOKENS,
        max_k_len: NUM_TOKENS,
    };
    let paged = F::paged_attention(&q, &cache, true).unwrap();
    assert_eq!(paged.shape(), vec![NUM_TOKENS, 1, HEAD_DIM]);

    // The cache now holds exactly the keys and values of this sequence, so plain attention over
    // the same tokens has to agree.
    let batched = |x: &Tensor| {
        x.view(&[1, NUM_TOKENS, 1, HEAD_DIM])
            .unwrap()
            .transpose(1, 2)
            .unwrap()
    };
    let dense = F::attention(&batched(&q), &batched(&k), &batched(&v), true).unwrap();

    let expected = to_host_f32(&dense);
    let actual = to_host_f32(&paged);
    assert!(
        expected
            .iter()
            .zip(&actual)
            .all(|(a, b)| (a - b).abs() < 1e-2),
        "paged and dense attention disagree"
    );
}
