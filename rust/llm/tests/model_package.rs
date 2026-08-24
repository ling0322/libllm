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

//! Reads a real model package, which the tests in `foundation.rs` cannot do with a file they
//! wrote themselves: whether this crate agrees with the C++ writer is only answered by a package
//! that writer produced.
//!
//! Model packages are large and are not in the repository, so this is `#[ignore]`d. Run it with
//! `cargo test --test model_package -- --ignored`, and point `LLM_TEST_PACKAGE` at a `.llmpkg`
//! if yours is not the one under `models/`.

use flint::{functional as F, Device};
use llm::{IniConfig, VarBuilder, ZipFile};

fn package_path() -> std::path::PathBuf {
    match std::env::var("LLM_TEST_PACKAGE") {
        Ok(path) => std::path::PathBuf::from(path),
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/llama3.2-3b-instruct-fp16.llmpkg"),
    }
}

#[test]
#[ignore = "needs a model package"]
fn reads_a_real_package() {
    let path = package_path();
    let package =
        ZipFile::open(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));

    let names = package.names();
    assert!(names.contains(&"model.ini"), "package holds {names:?}");
    assert!(names.contains(&"tokenizer.bin"), "package holds {names:?}");

    let config = IniConfig::parse(&package.read_to_string(llm::MODEL_CONFIG).unwrap()).unwrap();
    let model_type = config.section("model").unwrap().get_str("type").unwrap();
    let model_file = config
        .section("model")
        .unwrap()
        .get_str("model_file")
        .unwrap()
        .to_string();

    let section = config.section(model_type).unwrap();
    let hidden_size: i32 = section.get("hidden_size").unwrap();
    let vocab_size: i32 = section.get("vocab_size").unwrap();
    let num_layers: i32 = section.get("num_layers").unwrap();

    let float_type = F::default_float_type(Device::Cpu).unwrap();
    let vb = VarBuilder::from_reader(
        &mut package.open_entry(&model_file).unwrap(),
        Device::Cpu,
        float_type,
    )
    .unwrap();

    // Every layer the configuration promises is in the file, under the names the model looks for.
    assert!(vb.len() >= num_layers as usize, "{} tensors", vb.len());
    let model = vb.with_name(model_type);
    for layer in 0..num_layers {
        let block = model.with_name(&format!("block{layer}"));
        assert!(block.has("input_norm.weight"), "block{layer} is missing");
        assert!(block.has("attn.qkv_proj.weight"), "block{layer} is missing");
    }

    // The shapes the configuration calls for are the shapes the file holds.
    let embedding = llm::Embedding::build(hidden_size, vocab_size, &model.with_name("embd"))
        .expect("the embedding should match the configuration");
    let tokens = flint::Tensor::from_i64(&[3], &[0, 1, 2]).unwrap();
    let embedded = embedding.forward(&tokens).unwrap();
    assert_eq!(embedded.shape(), vec![3, hidden_size]);

    let norm = llm::RmsNorm::build(
        hidden_size,
        section.get("norm_eps").unwrap(),
        &model.with_name("norm"),
    )
    .unwrap();
    assert_eq!(
        norm.forward(&embedded).unwrap().shape(),
        vec![3, hidden_size]
    );
}
