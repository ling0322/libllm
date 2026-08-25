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

//! The tokenizer a package carries, which turns text into the ids a model reads.

use std::rc::Rc;

use crate::bpe::{BpeConfig, BpeEncoder, BpeModel};
use crate::error::{Error, Result};
use crate::zip_file::ZipFile;

/// A tokenizer and the vocabulary behind it.
///
/// Cloning shares the vocabulary, which is worth having: a model and the requests running against
/// it all encode with the same one.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    model: Rc<BpeModel>,
    config: BpeConfig,
}

impl Tokenizer {
    /// The entry of a package that says which tokenizer it holds.
    pub const CONFIG_FILE: &'static str = "tokenizer.ini";

    /// Read the tokenizer out of `package`.
    pub fn from_package(package: &ZipFile) -> Result<Tokenizer> {
        let ini = crate::ini::IniConfig::parse(&package.read_to_string(Self::CONFIG_FILE)?)?;
        let section = ini.section("tokenizer")?;

        match section.get_str("type")? {
            "bpe" => {
                let config = BpeConfig::from_section(section)?;
                let model = BpeModel::from_reader(&mut package.open_entry(&config.model_file)?)?;
                Ok(Tokenizer {
                    model: Rc::new(model),
                    config,
                })
            }
            other => Err(Error::model(format!(
                "unsupported tokenizer type {other:?}"
            ))),
        }
    }

    /// The ids of `text`.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        BpeEncoder::new(&self.model, &self.config).encode(text)
    }

    /// The vocabulary, for looking up control tokens and for turning ids back into text.
    pub fn vocab(&self) -> &BpeModel {
        &self.model
    }
}
