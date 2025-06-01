// src/bin/hello.rs
use onmi::Tensor;

fn main() {
    let x = Tensor::new((2, 3), onmi::DType::Float, onmi::Device::Cpu).unwrap();
    x.print();
}
