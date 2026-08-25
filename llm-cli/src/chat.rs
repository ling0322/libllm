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

//! The chat command: a conversation with a model, one question at a time.

use std::io::{BufRead, Write};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

use llm::{
    Device, Engine, EngineConfig, GenerationConfig, KVCacheManager, LlamaForGeneration, Message,
    RequestInput, RequestOutput, ZipFile,
};

use crate::args::{Args, DeviceOption};
use crate::download;

type Error = Box<dyn std::error::Error>;

/// Says what was wrong before printing the usage, which is the order the Go tool prints them in.
fn with_usage<T, E: std::fmt::Display>(result: Result<T, E>) -> Result<T, E> {
    if let Err(error) = &result {
        eprintln!("{error}\n");
        print_usage();
    }
    result
}

fn print_usage() {
    eprintln!("Usage: llm chat [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    crate::args::print_options();
    eprintln!();
}

pub fn main(arguments: &[String]) -> Result<(), Error> {
    let args = match Args::parse(arguments) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}\n");
            print_usage();
            return Err(error.into());
        }
    };
    if args.wants_help() {
        print_usage();
        return Ok(());
    }

    let model_name = with_usage(args.model())?;
    let device = with_usage(args.device())?;
    let model_path = download::model_path_or_download(model_name)?;

    let (engine, answers) = load(&model_path, device)?;

    println!("Please input your question.");
    println!("    Type ':new' to start a new session (clean history).");
    println!("    Type ':sys <system_prompt>' to set the system prompt and start a new session.");

    let mut history: Vec<Message> = Vec::new();
    let mut system_prompt = String::new();
    let mut turn = 0usize;

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("> ");
        std::io::stdout().flush()?;

        // End of input, which is how a session ends.
        let Some(line) = lines.next() else {
            println!();
            break;
        };
        let question = line?.trim().to_string();

        // A system prompt applies from a fresh conversation, so setting one clears the history.
        if let Some(prompt) = strip_prefix_ignoring_case(&question, ":sys ") {
            system_prompt = prompt.trim().to_string();
            history.clear();
            continue;
        }
        if question.eq_ignore_ascii_case(":new") {
            println!("===== new session =====");
            history.clear();
            continue;
        }
        if question.is_empty() {
            continue;
        }

        if history.is_empty() && !system_prompt.is_empty() {
            history.push(Message::new("system", system_prompt.clone()));
        }
        history.push(Message::new("user", question));

        turn += 1;
        let started = Instant::now();
        let (answer, num_tokens) = complete(&engine, &answers, &history, turn)?;
        let elapsed = started.elapsed().as_secs_f64();

        history.push(Message::new("assistant", answer));
        println!();
        let milliseconds_per_token = if num_tokens == 0 {
            0.0
        } else {
            elapsed * 1000.0 / num_tokens as f64
        };
        println!(
            "({num_tokens} tokens, time={elapsed:.2}s, \
             {milliseconds_per_token:.2}ms per token)"
        );
    }

    engine.shutdown()?;
    Ok(())
}

/// Loads the model and starts the engine that runs it.
///
/// The engine takes its callback once, at the start, so that callback feeds a channel and the
/// loop above reads whichever answer it is waiting for from there.
fn load(
    model_path: &std::path::Path,
    device: DeviceOption,
) -> Result<(Engine, Receiver<Chunk>), Error> {
    let device = match device {
        DeviceOption::Cpu => Device::Cpu,
        DeviceOption::Cuda => Device::Cuda,
        DeviceOption::Auto => {
            if Device::Cuda.is_available() {
                Device::Cuda
            } else {
                Device::Cpu
            }
        }
    };

    // The attention this model needs has CUDA kernels only, and a missing kernel ends the process
    // rather than reporting anything, so it is worth saying now.
    if device == Device::Cpu {
        return Err(
            "this model needs a CUDA device: the paged attention kernels it runs on have \
                    no CPU implementation yet"
                .into(),
        );
    }

    let path = model_path.to_path_buf();
    let config = EngineConfig::default();
    let (chunks, arriving) = channel::<Chunk>();

    let engine = Engine::new(
        move || {
            let package = ZipFile::open(&path)?;
            let model = LlamaForGeneration::from_package(device, &package)?;
            let cache = KVCacheManager::for_model(&model, &config)?;
            Ok((model, cache))
        },
        config.max_num_batched_tokens,
        make_callback(chunks),
    )?;

    Ok((engine, arriving))
}

/// What the model produced, as it arrives.
enum Chunk {
    Text(String),
    Done(Option<String>),
}

/// Asks one question and prints the answer as it comes, returning it and how many tokens it took.
fn complete(
    engine: &Engine,
    answers: &Receiver<Chunk>,
    history: &[Message],
    turn: usize,
) -> Result<(String, usize), Error> {
    let request_id = format!("chat-{turn}");

    engine.add_request_input(
        &request_id,
        RequestInput::Messages(history.to_vec()),
        GenerationConfig::default(),
    )?;

    let mut answer = String::new();
    let mut num_tokens = 0;
    // One question is in flight at a time, so whatever arrives belongs to this one.
    for chunk in answers {
        match chunk {
            Chunk::Text(text) => {
                print!("{text}");
                std::io::stdout().flush()?;
                answer.push_str(&text);
                num_tokens += 1;
            }
            Chunk::Done(None) => break,
            Chunk::Done(Some(error)) => return Err(error.into()),
        }
    }

    Ok((answer, num_tokens))
}

/// Turns the engine's outputs into the chunks the loop above reads.
fn make_callback(chunks: Sender<Chunk>) -> impl Fn(&[RequestOutput]) + Send + 'static {
    move |outputs: &[RequestOutput]| {
        for output in outputs {
            if !output.text.is_empty() {
                let _ = chunks.send(Chunk::Text(output.text.clone()));
            }
            if output.finished {
                let failed =
                    (!output.error_message.is_empty()).then(|| output.error_message.clone());
                let _ = chunks.send(Chunk::Done(failed));
            }
        }
    }
}

/// `text` without `prefix`, comparing the prefix without regard to case.
fn strip_prefix_ignoring_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if text.len() >= prefix.len() && text[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&text[prefix.len()..])
    } else {
        None
    }
}
