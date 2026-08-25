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

//! Finding a model: in the cache if it is there, from the network if it is not.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::cli::args::Args;

type Error = Box<dyn std::error::Error>;

/// The models this tool knows how to fetch, as name, URL and file name.
const MODELS: &[(&str, &str, &str)] = &[(
    "llama3.2:3b:fp16",
    "https://huggingface.co/ling0322/llama3.2-libllm/resolve/main/\
     llama3.2-3b-instruct-fp16.llmpkg",
    "llama3.2-3b-instruct-fp16.llmpkg",
)];

/// The shorter names people type, and what they mean.
const ALIASES: &[(&str, &str)] = &[
    ("llama3.2", "llama3.2:3b:fp16"),
    ("llama3.2:3b", "llama3.2:3b:fp16"),
    ("llama3.2:3b:fp16", "llama3.2:3b:fp16"),
];

/// Says what was wrong before printing the usage, which is the order the Go tool prints them in.
fn with_usage<T, E: std::fmt::Display>(result: Result<T, E>) -> Result<T, E> {
    if let Err(error) = &result {
        eprintln!("{error}\n");
        print_usage();
    }
    result
}

fn print_usage() {
    eprintln!("Usage: llm download [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    crate::cli::args::print_options();
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

    let name = with_usage(args.model())?;
    if let Some(path) = cached(name)? {
        println!(
            "model \"{name}\" already downloaded. Path is \"{}\"",
            path.display()
        );
    }

    download(name)?;
    Ok(())
}

/// The path of a model, downloading it first if it is not here yet.
///
/// A name that ends in `.llmpkg` is a file the caller already has; anything else is a name from
/// the table above.
pub fn model_path_or_download(name_or_path: &str) -> Result<PathBuf, Error> {
    let path = if Path::new(name_or_path)
        .extension()
        .is_some_and(|ext| ext == "llmpkg")
    {
        PathBuf::from(name_or_path)
    } else {
        match cached(name_or_path)? {
            Some(path) => path,
            None => download(name_or_path)?,
        }
    };

    if !path.exists() {
        return Err(format!("model file \"{}\" does not exist", path.display()).into());
    }
    Ok(path)
}

/// The full name of a model, as the table knows it.
fn resolve(name: &str) -> Result<&'static str, Error> {
    ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map(|(_, full)| *full)
        .ok_or_else(|| format!("unable to resolve model name \"{name}\"").into())
}

fn entry(name: &str) -> Result<(&'static str, &'static str), Error> {
    let resolved = resolve(name)?;
    MODELS
        .iter()
        .find(|(model, _, _)| *model == resolved)
        .map(|(_, url, filename)| (*url, *filename))
        .ok_or_else(|| format!("invalid model name \"{name}\"").into())
}

/// Where downloaded models are kept: under the user's home on unix, next to the executable on
/// Windows, where a home directory is not where a program's data belongs.
fn cache_dir() -> Result<PathBuf, Error> {
    if cfg!(windows) {
        let executable = std::env::current_exe()?;
        let directory = executable
            .parent()
            .ok_or("unable to find the directory of the executable")?;
        Ok(directory.join("models"))
    } else {
        let home = std::env::var("HOME").map_err(|_| "unable to find the home directory")?;
        Ok(PathBuf::from(home).join(".libllm").join("models"))
    }
}

/// The path of a model that is already downloaded.
fn cached(name: &str) -> Result<Option<PathBuf>, Error> {
    let (_, filename) = entry(name)?;
    let path = cache_dir()?.join(filename);
    Ok(path.exists().then_some(path))
}

/// Fetches a model into the cache and returns where it landed.
fn download(name: &str) -> Result<PathBuf, Error> {
    let (url, filename) = entry(name)?;
    let path = cache_dir()?.join(filename);
    std::fs::create_dir_all(path.parent().expect("the cache path has a directory"))?;

    // Downloaded under another name and moved into place at the end, so that an interrupted
    // download is never mistaken for a model.
    let partial = path.with_extension("llmpkg.download");
    fetch(url, &partial, filename)?;
    std::fs::rename(&partial, &path)?;

    println!("saved to {}", path.display());
    Ok(path)
}

/// Reads `url` into `path`, showing how far along it is.
fn fetch(url: &str, path: &Path, filename: &str) -> Result<(), Error> {
    let response = ureq::get(url).call()?;
    let total: u64 = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    let mut body = response.into_body().into_reader();
    let mut file = std::fs::File::create(path)?;

    let mut progress = Progress::new(filename, total);
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = body.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        progress.advance(read as u64);
    }
    progress.finish();

    Ok(())
}

/// A progress bar, drawn over itself on one line.
struct Progress {
    filename: String,
    total: u64,
    done: u64,
    started: Instant,
    last_drawn: Instant,
}

impl Progress {
    fn new(filename: &str, total: u64) -> Progress {
        let now = Instant::now();
        Progress {
            filename: filename.to_string(),
            total,
            done: 0,
            started: now,
            last_drawn: now - std::time::Duration::from_secs(1),
        }
    }

    fn advance(&mut self, bytes: u64) {
        self.done += bytes;
        // Drawing on every block would spend more time on the terminal than on the download.
        if self.last_drawn.elapsed().as_millis() >= 100 {
            self.draw();
            self.last_drawn = Instant::now();
        }
    }

    fn draw(&self) {
        let seconds = self.started.elapsed().as_secs_f64().max(1e-6);
        let rate = self.done as f64 / seconds;

        let bar = if self.total > 0 {
            let fraction = (self.done as f64 / self.total as f64).clamp(0.0, 1.0);
            let filled = (fraction * 30.0).round() as usize;
            format!(
                "{:>3}% |{}{}| {} / {}",
                (fraction * 100.0).round() as u64,
                "█".repeat(filled),
                " ".repeat(30 - filled),
                bytes(self.done),
                bytes(self.total)
            )
        } else {
            // Without a content length there is no bar to fill, only a count.
            bytes(self.done)
        };

        eprint!("\r{} {} [{}/s]   ", self.filename, bar, bytes(rate as u64));
        let _ = std::io::stderr().flush();
    }

    fn finish(&self) {
        self.draw();
        eprintln!();
    }
}

/// A byte count as a person would write it.
fn bytes(count: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];

    let mut value = count as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{count} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_the_names_people_type() {
        assert_eq!(resolve("llama3.2").unwrap(), "llama3.2:3b:fp16");
        assert_eq!(resolve("llama3.2:3b").unwrap(), "llama3.2:3b:fp16");
        assert!(resolve("llama9").is_err());

        let (url, filename) = entry("llama3.2").unwrap();
        assert!(url.starts_with("https://"), "{url}");
        assert_eq!(filename, "llama3.2-3b-instruct-fp16.llmpkg");
    }

    #[test]
    fn every_model_in_the_table_can_be_named() {
        // An entry nobody can name is unreachable, and an alias for nothing is a broken name.
        for (name, _, _) in MODELS {
            assert!(ALIASES.iter().any(|(_, full)| full == name), "{name}");
        }
        for (alias, full) in ALIASES {
            assert!(
                MODELS.iter().any(|(name, _, _)| name == full),
                "{alias} names nothing"
            );
        }
    }

    #[test]
    fn a_file_path_is_taken_as_it_is() {
        // A path that does not exist is reported rather than looked up as a model name.
        let error = model_path_or_download("/nowhere/model.llmpkg")
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not exist"), "{error}");
    }

    #[test]
    fn writes_byte_counts_the_way_a_person_reads_them() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1024), "1.0 KB");
        assert_eq!(bytes(1024 * 1024 * 3 / 2), "1.5 MB");
        assert_eq!(bytes(7_283_348_871), "6.8 GB");
    }
}
