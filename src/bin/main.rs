// src/bin/hello.rs
use llm::{Tensor, F};

fn main() -> llm::Result<()> {
    let x: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from_slice((2, 3), x)?;
    let x = x.to_device(llm::Device::Cuda)?;
    let x = x.to_dtype(llm::DType::Float16)?;
    let x = F::softmax(&x)?;

    x.print();
    Ok(())
}
