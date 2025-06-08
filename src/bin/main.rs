// src/bin/hello.rs
use onmi::Tensor;

fn main() {
    let x = Tensor::new((2, 3), onmi::DType::Float, onmi::Device::Cpu).unwrap();
    let x = x.slice(0, 1..).unwrap();
    x.print();

    let d = x.data::<i64>().unwrap();
    println!("{}", d[0])
}
