// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Prints what a model package holds: its entries, its configuration, and the tensors its
//! parameter file stores.
//!
//! ```text
//! cargo run --release --example inspect -- models/llama3.2-3b-instruct-fp16.llmpkg
//! ```

use llm::flint::{functional as F, Device};

fn main() -> Result<(), llm::Error> {
    let path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("usage: inspect <package.llmpkg>");
            std::process::exit(2);
        }
    };

    let package = llm::ZipFile::open(&path)?;
    println!("entries: {:?}", package.names());

    let config = llm::IniConfig::parse(&package.read_to_string(llm::MODEL_CONFIG)?)?;
    let model = config.section("model")?;
    let model_type = model.get_str("type")?.to_string();
    let model_file = model.get_str("model_file")?.to_string();
    println!("model: type={model_type} file={model_file}");

    let start = std::time::Instant::now();
    let vb = llm::VarBuilder::from_reader(
        &mut package.open_entry(&model_file)?,
        Device::Cpu,
        F::default_float_type(Device::Cpu)?,
    )?;
    println!("{} tensors read in {:?}", vb.len(), start.elapsed());

    for name in vb.names() {
        println!("  {name}");
    }

    Ok(())
}
