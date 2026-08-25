//! Tests for the operations in `flint::functional`. Like the tensor tests, these link against the
//! shared library that CMake builds, so run `cmake --build build --target libllm` first.
//!
//! Only the operations the CPU backend has kernels for are exercised here, and only with shapes
//! they accept: the operators check their arguments with the C++ library's fatal check, so a test
//! that fed one a shape it cannot use would abort the whole test binary rather than fail.

use flint::{functional as F, DType, Device, Tensor};

fn cpu_f32(shape: &[i32], data: &[f32]) -> Tensor {
    Tensor::from_f32(shape, data).unwrap()
}

#[test]
fn does_element_wise_arithmetic() {
    let a = cpu_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = cpu_f32(&[2, 2], &[10.0, 20.0, 30.0, 40.0]);

    assert_eq!(
        F::add(&a, &b).unwrap().to_vec_f32().unwrap(),
        vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(
        F::sub(&b, &a).unwrap().to_vec_f32().unwrap(),
        vec![9.0, 18.0, 27.0, 36.0]
    );
    assert_eq!(
        F::mul(&a, &b).unwrap().to_vec_f32().unwrap(),
        vec![10.0, 40.0, 90.0, 160.0]
    );
    assert_eq!(
        F::mul_scalar(&a, 2.0).unwrap().to_vec_f32().unwrap(),
        vec![2.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn broadcasts_the_trailing_dimensions() {
    let a = cpu_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let row = cpu_f32(&[3], &[1.0, 0.0, -1.0]);

    assert_eq!(
        F::add(&a, &row).unwrap().to_vec_f32().unwrap(),
        vec![2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
    );
}

#[test]
fn multiplies_matrices() {
    let a = cpu_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = cpu_f32(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let c = F::matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.to_vec_f32().unwrap(), vec![4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn reduces_over_the_last_dimension() {
    let x = cpu_f32(&[2, 3], &[1.0, 2.0, 3.0, 40.0, 50.0, 60.0]);

    let sums = F::sum(&x, F::LAST_DIM).unwrap();
    assert_eq!(sums.shape(), vec![2]);
    assert_eq!(sums.to_vec_f32().unwrap(), vec![6.0, 150.0]);
    assert_eq!(
        F::max(&x, F::LAST_DIM).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 60.0]
    );
}

#[test]
fn softmax_gives_rows_that_sum_to_one() {
    let x = cpu_f32(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]);
    let probabilities = F::softmax(&x).unwrap();

    let sums = F::sum(&probabilities, F::LAST_DIM).unwrap();
    let ones = cpu_f32(&[2], &[1.0, 1.0]);
    assert!(F::all_close(&sums, &ones).unwrap());

    // A uniform row stays uniform.
    let values = probabilities.to_vec_f32().unwrap();
    assert!(values[3..].iter().all(|p| (p - 1.0 / 3.0).abs() < 1e-5));
}

#[test]
fn halves_the_last_dimension_with_swiglu() {
    // swish(x) = x * sigmoid(x), so the gate at 0 kills the first output.
    let x = cpu_f32(&[1, 4], &[0.0, 1.0, 3.0, 5.0]);
    let y = F::swiglu(&x).unwrap();

    assert_eq!(y.shape(), vec![1, 2]);
    let values = y.to_vec_f32().unwrap();
    let swish_one = 1.0 / (1.0 + (-1.0f32).exp());
    assert!(values[0].abs() < 1e-6, "unexpected output: {values:?}");
    assert!(
        (values[1] - swish_one * 5.0).abs() < 1e-5,
        "unexpected output: {values:?}"
    );
}

#[test]
fn normalizes_with_rms_norm() {
    let x = cpu_f32(&[1, 4], &[1.0, 1.0, 1.0, 1.0]);
    let weight = cpu_f32(&[4], &[2.0, 2.0, 2.0, 2.0]);

    // The root mean square of a row of ones is one, so only the weight is left.
    let y = F::rms_norm(&x, &weight, 1e-5).unwrap();
    assert!(F::all_close(&y, &cpu_f32(&[1, 4], &[2.0; 4])).unwrap());
}

#[test]
fn looks_up_embedding_rows() {
    let table = cpu_f32(&[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]);
    let indices = Tensor::from_i64(&[2], &[2, 0]).unwrap();

    let rows = F::lookup(&table, &indices).unwrap();
    assert_eq!(rows.shape(), vec![2, 2]);
    assert!(F::all_close(&rows, &cpu_f32(&[2, 2], &[2.0, 2.1, 0.0, 0.1])).unwrap());
}

#[test]
fn concatenates_along_either_dimension() {
    let a = cpu_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = cpu_f32(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);

    let rows = F::cat(&a, &b, 0).unwrap();
    assert_eq!(rows.shape(), vec![4, 2]);
    assert_eq!(
        rows.to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );

    let columns = F::cat(&a, &b, 1).unwrap();
    assert_eq!(columns.shape(), vec![2, 4]);
    assert_eq!(
        columns.to_vec_f32().unwrap(),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );
}

#[test]
fn builds_a_causal_mask() {
    let mask = F::causal_mask(3, Device::Cpu).unwrap();
    assert_eq!(mask.shape(), vec![3, 3]);

    let values = mask.cast(DType::Float).unwrap().to_vec_f32().unwrap();
    // The lower triangle may attend and the upper one may not.
    assert_eq!(values[0], 0.0);
    assert_eq!(values[3], 0.0);
    assert!(
        values[1] < 0.0 && values[1].is_infinite(),
        "unexpected mask: {values:?}"
    );
}

#[test]
fn writes_in_place() {
    let mut x = Tensor::zeros(&[2, 2], DType::Float, Device::Cpu).unwrap();
    F::fill(&mut x, 3.0).unwrap();
    assert_eq!(x.to_vec_f32().unwrap(), vec![3.0; 4]);

    let source = cpu_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    F::copy(&source, &mut x).unwrap();
    assert_eq!(x.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn in_place_writes_are_visible_through_a_view() {
    let mut x = Tensor::zeros(&[2, 2], DType::Float, Device::Cpu).unwrap();
    let row = x.subtensor(0).unwrap();

    F::fill(&mut x, 5.0).unwrap();

    // The view shares the storage, so it sees the fill.
    assert_eq!(row.to_vec_f32().unwrap(), vec![5.0, 5.0]);
}

#[test]
fn penalizes_repeated_tokens() {
    let mut logits = cpu_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
    let history = Tensor::from_i64(&[1, 2], &[1, 3]).unwrap();

    F::repetition_penalty(&mut logits, &history, 2.0).unwrap();

    let values = logits.to_vec_f32().unwrap();
    assert_eq!(values[0], 1.0, "an unseen token keeps its logit");
    assert_eq!(values[2], 3.0, "an unseen token keeps its logit");
    assert!(values[1] < 2.0, "a repeated token loses ground: {values:?}");
    assert!(values[3] < 4.0, "a repeated token loses ground: {values:?}");
}

#[test]
fn samples_greedily_at_temperature_zero() {
    let logits = cpu_f32(&[2, 3], &[1.0, 9.0, 1.0, 5.0, 1.0, 1.0]);
    let temperatures = cpu_f32(&[2], &[0.0, 0.0]);
    let top_ks = Tensor::from_i32(&[2], &[0, 0]).unwrap();
    let top_ps = cpu_f32(&[2], &[1.0, 1.0]);

    let labels = F::sample_with_params(&logits, &temperatures, &top_ks, &top_ps).unwrap();
    assert_eq!(labels.shape(), vec![2]);
    assert_eq!(labels.to_vec_i64().unwrap(), vec![1, 0]);
}

#[test]
fn sampling_with_top_k_of_one_ignores_the_temperature() {
    let logits = cpu_f32(&[1, 3], &[1.0, 1.0, 9.0]);
    let temperatures = cpu_f32(&[1], &[10.0]);
    let top_ks = Tensor::from_i32(&[1], &[1]).unwrap();
    let top_ps = cpu_f32(&[1], &[1.0]);

    let labels = F::sample_with_params(&logits, &temperatures, &top_ks, &top_ps).unwrap();
    assert_eq!(labels.to_vec_i64().unwrap(), vec![2]);
}

#[test]
fn draws_reproducible_random_numbers() {
    // The generator is shared by everything on the device, so this has to stay the only test in
    // the file that draws from it, or another one running alongside will move it between the draws.
    F::manual_seed(Device::Cpu, 42).unwrap();
    let first = F::rand(&[8], DType::Float, Device::Cpu).unwrap();

    F::manual_seed(Device::Cpu, 42).unwrap();
    let second = F::rand(&[8], DType::Float, Device::Cpu).unwrap();

    assert_eq!(first.to_vec_f32().unwrap(), second.to_vec_f32().unwrap());
    assert!(
        first
            .to_vec_f32()
            .unwrap()
            .iter()
            .all(|x| (0.0..1.0).contains(x)),
        "uniform numbers should land in [0, 1)"
    );
}

#[test]
fn compares_within_a_tolerance() {
    let a = cpu_f32(&[3], &[1.0, 2.0, 3.0]);
    let b = cpu_f32(&[3], &[1.0, 2.0, 3.0001]);

    assert!(F::all_close(&a, &b).unwrap());
    assert!(!F::all_close_with_tolerance(&a, &b, 0.0, 1e-9).unwrap());
}

#[test]
fn reports_the_default_float_type() {
    assert_eq!(F::default_float_type(Device::Cpu).unwrap(), DType::Float);
}

#[test]
fn prints_without_failing() {
    let x = cpu_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    F::print(&x).unwrap();
}
