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

//! Tests for the model package formats: the stored-zip archive, `model.ini`, and the parameter
//! file, plus the layers built out of one.
//!
//! Every format is written here by hand and read back, so the tests say what this crate believes
//! the formats are. `tests/model_package.rs` checks that belief against a real package.

use std::io::Write;

use flint::{functional as F, DType, Device, Tensor};
use llm::{Embedding, IniConfig, Linear, RmsNorm, VarBuilder, ZipFile};

/// Writes a zip holding `entries`, stored rather than compressed, as a model package is.
fn write_package(path: &std::path::Path, entries: &[(&str, &[u8])]) {
    let mut file = std::fs::File::create(path).unwrap();
    let mut directory = Vec::new();
    let mut offset = 0u32;

    for (name, data) in entries {
        let mut header = Vec::new();
        header.extend_from_slice(&0x0403_4b50u32.to_le_bytes()); // signature
        header.extend_from_slice(&[10, 0]); // version
        header.extend_from_slice(&[0, 0]); // flag
        header.extend_from_slice(&[0, 0]); // compression: stored
        header.extend_from_slice(&[0, 0, 0, 0]); // modification time and date
        header.extend_from_slice(&[0, 0, 0, 0]); // crc32, which nothing here checks
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());
        header.extend_from_slice(&(name.len() as u16).to_le_bytes());
        header.extend_from_slice(&[0, 0]); // extra field length
        header.extend_from_slice(name.as_bytes());

        let local_offset = offset;
        file.write_all(&header).unwrap();
        file.write_all(data).unwrap();
        offset += (header.len() + data.len()) as u32;

        directory.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        directory.extend_from_slice(&[10, 0, 10, 0]); // version made by, version needed
        directory.extend_from_slice(&[0, 0, 0, 0]); // flag, compression
        directory.extend_from_slice(&[0, 0, 0, 0]); // modification time and date
        directory.extend_from_slice(&[0, 0, 0, 0]); // crc32
        directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
        directory.extend_from_slice(&[0, 0, 0, 0]); // extra field and comment lengths
        directory.extend_from_slice(&[0, 0, 0, 0]); // start disk, internal attributes
        directory.extend_from_slice(&[0, 0, 0, 0]); // external attributes
        directory.extend_from_slice(&local_offset.to_le_bytes());
        directory.extend_from_slice(name.as_bytes());
    }

    let directory_offset = offset;
    file.write_all(&directory).unwrap();

    let mut end = Vec::new();
    end.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    end.extend_from_slice(&[0, 0, 0, 0]); // disk numbers
    end.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    end.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    end.extend_from_slice(&(directory.len() as u32).to_le_bytes());
    end.extend_from_slice(&directory_offset.to_le_bytes());
    end.extend_from_slice(&[0, 0]); // comment length
    file.write_all(&end).unwrap();
}

/// Writes one tensor in the form a parameter file holds it.
fn write_tensor(out: &mut Vec<u8>, shape: &[i32], dtype: DType, data: &[u8]) {
    out.extend_from_slice(b"tnsr");
    out.extend_from_slice(&(shape.len() as i16).to_le_bytes());
    for size in shape {
        out.extend_from_slice(&size.to_le_bytes());
    }

    let numel: i64 = shape.iter().map(|&size| size as i64).product();
    out.extend_from_slice(b"tdat");
    out.extend_from_slice(&1i32.to_le_bytes()); // one slot
    out.extend_from_slice(&(dtype.code() as i16).to_le_bytes());
    out.extend_from_slice(&numel.to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&0x55aai16.to_le_bytes());
}

/// Writes a parameter file holding `tensors`, each given as a name, a shape, and its elements.
fn write_params(tensors: &[(&str, &[i32], &[f32])]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"llyn::tdicv2    ");
    out.extend_from_slice(b"<d> ");

    for (name, shape, values) in tensors {
        out.extend_from_slice(b"<r> ");
        out.extend_from_slice(&(name.len() as i16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());

        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        write_tensor(&mut out, shape, DType::Float, &bytes);
        out.extend_from_slice(b"</r>");
    }

    out.extend_from_slice(b"</d>");
    out
}

fn cpu_builder(tensors: &[(&str, &[i32], &[f32])]) -> VarBuilder {
    let params = write_params(tensors);
    VarBuilder::from_reader(&mut &params[..], Device::Cpu, DType::Float).unwrap()
}

#[test]
fn reads_a_stored_package() {
    let dir = std::env::temp_dir().join(format!("llm-package-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.llmpkg");

    write_package(
        &path,
        &[
            ("model.ini", b"[model]\ntype = llama\n"),
            ("model.bin", &[1, 2, 3, 4]),
        ],
    );

    let package = ZipFile::open(&path).unwrap();
    assert_eq!(package.names(), vec!["model.bin", "model.ini"]);
    assert!(package.contains("model.ini"));
    assert!(!package.contains("tokenizer.bin"));
    assert_eq!(package.read("model.bin").unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(
        package.read_to_string("model.ini").unwrap(),
        "[model]\ntype = llama\n"
    );

    // An entry stops at its own end rather than running into the next one.
    let error = package.read("nothing").unwrap_err();
    assert!(error.to_string().contains("nothing"), "{error}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn reads_a_configuration() {
    let config = IniConfig::parse(
        "; the model this package holds\n\
         [llama]\n\
         hidden_size = 3072\n\
         norm_eps = 1e-05\n\
         qkv_proj_bias = false  # off for llama\n\
         \n\
         [model]\n\
         type = llama\n",
    )
    .unwrap();

    let llama = config.section("llama").unwrap();
    assert_eq!(llama.get::<i32>("hidden_size").unwrap(), 3072);
    assert_eq!(llama.get::<f32>("norm_eps").unwrap(), 1e-5);
    assert!(!llama.get_bool("qkv_proj_bias").unwrap());
    assert_eq!(
        config.section("model").unwrap().get_str("type").unwrap(),
        "llama"
    );

    // A key a model only writes down when it departs from the usual.
    assert_eq!(llama.get_or("num_layers", 28).unwrap(), 28);
    assert!(llama.get_bool_or("tie_embeddings", true).unwrap());

    assert!(!config.has_section("qwen"));
    assert!(config.section("qwen").is_err());
    assert!(llama.get::<i32>("norm_eps").is_err(), "1e-05 is not an int");
    assert!(llama.get_str("missing").is_err());
}

#[test]
fn reads_parameters_and_their_namespaces() {
    let vb = cpu_builder(&[
        (
            "llama.embd.weight",
            &[3, 2],
            &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
        ),
        ("llama.norm.weight", &[2], &[1.0, 1.0]),
    ]);

    assert_eq!(vb.len(), 2);
    assert_eq!(vb.names(), vec!["llama.embd.weight", "llama.norm.weight"]);

    let embd = vb.with_name("llama").with_name("embd");
    assert_eq!(embd.name(), "llama.embd");
    assert!(embd.has("weight"));
    assert!(!embd.has("bias"));

    let weight = embd.get("weight", &[3, 2]).unwrap();
    assert_eq!(weight.shape(), vec![3, 2]);
    assert_eq!(
        weight.to_vec_f32().unwrap(),
        vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1]
    );

    // The shape the caller expects is checked, since the alternative is a failure much later.
    let error = embd.get("weight", &[2, 3]).unwrap_err();
    assert!(error.to_string().contains("shape"), "{error}");

    let error = embd.get("bias", &[2]).unwrap_err();
    assert!(error.to_string().contains("llama.embd.bias"), "{error}");
}

#[test]
fn refuses_a_parameter_file_it_does_not_understand() {
    let mut params = write_params(&[("weight", &[2], &[1.0, 2.0])]);

    let truncated = &params[..params.len() - 8];
    assert!(VarBuilder::from_reader(&mut &truncated[..], Device::Cpu, DType::Float).is_err());

    params[0] = b'x';
    let error = VarBuilder::from_reader(&mut &params[..], Device::Cpu, DType::Float).unwrap_err();
    assert!(error.to_string().contains("tag"), "{error}");
}

#[test]
fn builds_and_runs_the_layers() {
    let vb = cpu_builder(&[
        ("embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
        ("norm.weight", &[2], &[2.0, 2.0]),
        ("proj.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("proj.bias", &[2], &[0.5, -0.5]),
    ]);

    let embedding = Embedding::build(2, 3, &vb.with_name("embd")).unwrap();
    let tokens = Tensor::from_i64(&[2], &[2, 0]).unwrap();
    let embedded = embedding.forward(&tokens).unwrap();
    assert_eq!(embedded.shape(), vec![2, 2]);
    assert!(F::all_close(
        &embedded,
        &Tensor::from_f32(&[2, 2], &[2.0, 2.1, 0.0, 0.1]).unwrap()
    )
    .unwrap());

    // The root mean square of a row of ones is one, so the weight is what is left.
    let norm = RmsNorm::build(2, 1e-5, &vb.with_name("norm")).unwrap();
    let normed = norm
        .forward(&Tensor::from_f32(&[1, 2], &[1.0, 1.0]).unwrap())
        .unwrap();
    assert!(F::all_close(&normed, &Tensor::from_f32(&[1, 2], &[2.0, 2.0]).unwrap()).unwrap());

    let linear = Linear::build(2, 2, true, &vb.with_name("proj")).unwrap();
    let projected = linear
        .forward(&Tensor::from_f32(&[1, 2], &[1.0, 2.0]).unwrap())
        .unwrap();
    assert_eq!(projected.to_vec_f32().unwrap(), vec![1.5, 1.5]);

    // Weights the model does not expect mean the two disagree about what the layer is.
    let error = Linear::build(2, 2, false, &vb.with_name("proj")).unwrap_err();
    assert!(error.to_string().contains("bias"), "{error}");
}

#[test]
fn reports_a_wrongly_shaped_input_instead_of_ending_the_process() {
    let vb = cpu_builder(&[("embd.weight", &[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1])]);
    let embedding = Embedding::build(2, 3, &vb.with_name("embd")).unwrap();

    // Packed token ids are 1-D. The tensor library would abort on this, so it is caught here.
    let tokens = Tensor::from_i64(&[1, 2], &[2, 0]).unwrap();
    let error = embedding.forward(&tokens).unwrap_err();
    assert!(error.to_string().contains("2-D"), "{error}");
}
