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

//! Language model inference on top of the flint tensor library.
//!
//! This is a port of the C++ `src/libllm` to Rust. It reads a model package, builds the model it
//! describes, and runs it; the tensor operations themselves are the ones [`flint`] binds, and
//! that module is the safe wrapper over the native `libflint.a` this crate links.
//!
//! ```no_run
//! use llm::{DType, Device, VarBuilder, ZipFile};
//!
//! let mut package = ZipFile::open("model.llmpkg")?;
//! let config = llm::IniConfig::parse(&package.read_to_string(llm::MODEL_CONFIG)?)?;
//! let mut params = package.open_entry(config.section("model")?.get_str("model_file")?)?;
//! let vb = VarBuilder::from_reader(&mut params, Device::Cpu, DType::Float)?;
//! println!("{} tensors", vb.len());
//! # Ok::<(), llm::Error>(())
//! ```
//!
//! # Threading
//!
//! A [`flint::Tensor`] stays on the thread that made it, so everything built out of one does too.

mod bpe;
pub mod capi;
mod engine;
mod engine_config;
mod error;
pub mod flint;
mod forward_batch;
mod ini;
mod kv_cache;
mod layers;
mod llama;
mod model;
mod prompt;
mod reader;
mod request;
mod sampling_batch;
mod scheduler;
mod tokenizer;
mod var_builder;
mod zip_file;

pub use bpe::{BpeConfig, BpeEncoder, BpeModel, INVALID_TOKEN};
/// The tensor types a caller of this crate needs to name, re-exported so that the common case
/// does not have to reach into [`flint`].
pub use flint::{DType, Device};

pub use engine::{Engine, RequestInput};
pub use engine_config::EngineConfig;
pub use error::{Error, Result};
pub use forward_batch::{ForwardBatch, PreparedBatch};
pub use ini::{IniConfig, IniSection};
pub use kv_cache::{KVCacheManager, KVCacheSpec};
pub use layers::{Embedding, Linear, RmsNorm};
pub use llama::{LlamaConfig, LlamaForGeneration, LlamaModel};
pub use model::ModelForGeneration;
pub use prompt::{Message, Prompt, PromptBlock};
pub use reader::BinaryRead;
pub use request::{FinishReason, GenerationConfig, Request, RequestOutput, RequestStatus};
pub use sampling_batch::{PreparedSampling, SamplingBatch};
pub use scheduler::Scheduler;
pub use tokenizer::Tokenizer;
pub use var_builder::VarBuilder;
pub use zip_file::ZipFile;

/// The name of the configuration entry every model package holds.
pub const MODEL_CONFIG: &str = "model.ini";
