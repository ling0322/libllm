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

//! Answers one question with a model package, end to end.
//!
//! ```text
//! cargo run --release --example chat -- models/llama3.2-3b-instruct-fp16.llmpkg "Why is the sky blue?"
//! ```

use std::io::Write;
use std::sync::mpsc::channel;

use flint::Device;
use llm::{
    Engine, EngineConfig, GenerationConfig, KVCacheManager, LlamaForGeneration, Message,
    RequestInput, RequestOutput, ZipFile,
};

fn main() -> Result<(), llm::Error> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(question)) = (args.next(), args.next()) else {
        eprintln!("usage: chat <package.llmpkg> <question>");
        std::process::exit(2);
    };

    // The paged attention kernels a model needs are CUDA only today.
    let device = if Device::Cuda.is_available() {
        Device::Cuda
    } else {
        eprintln!("this needs a CUDA device");
        std::process::exit(1);
    };

    // Tells the main thread when the answer is complete, since generation runs elsewhere.
    let (finished_tx, finished_rx) = channel();

    let config = EngineConfig::default();
    let engine = Engine::new(
        move || {
            let package = ZipFile::open(&path)?;
            let model = LlamaForGeneration::from_package(device, &package)?;
            let cache = KVCacheManager::for_model(&model, &config)?;
            Ok((model, cache))
        },
        config.max_num_batched_tokens,
        move |outputs: &[RequestOutput]| {
            for output in outputs {
                print!("{}", output.text);
                let _ = std::io::stdout().flush();
                if output.finished {
                    println!("\n[{:?}] {}", output.finish_reason, output.error_message);
                    let _ = finished_tx.send(());
                }
            }
        },
    )?;

    engine.add_request_input(
        "chat",
        RequestInput::Messages(vec![Message::new("user", question)]),
        GenerationConfig {
            temperature: 0.0,
            max_tokens: 256,
            ..GenerationConfig::default()
        },
    )?;

    let _ = finished_rx.recv();
    engine.shutdown()
}
