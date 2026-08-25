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

//! Reading `model.ini`, the configuration that says which model a package holds and how it is
//! shaped.
//!
//! The format is the usual one: `[section]` headers, `key = value` lines, and `;` or `#` comments.
//! Keys are looked up per section, and asking for one that is missing or that does not parse is
//! an error rather than a default, since a model built on a guessed hyperparameter would only
//! fail later and less clearly.

use std::collections::BTreeMap;
use std::str::FromStr;

use crate::error::{Error, Result};

/// One `[section]` of the configuration.
#[derive(Clone, Debug, Default)]
pub struct IniSection {
    name: String,
    entries: BTreeMap<String, String>,
}

impl IniSection {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the section has a `key =` line.
    pub fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// The value of `key` as written.
    pub fn get_str(&self, key: &str) -> Result<&str> {
        self.entries
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| Error::format(format!("{}: key {key:?} not found", self.name)))
    }

    /// The value of `key`, parsed. Used for the numeric hyperparameters.
    pub fn get<T: FromStr>(&self, key: &str) -> Result<T> {
        let value = self.get_str(key)?;
        value.parse().map_err(|_| {
            Error::format(format!(
                "{}: key {key:?} holds {value:?}, which is not a {}",
                self.name,
                std::any::type_name::<T>()
            ))
        })
    }

    /// The value of `key`, parsed, or `default` when the key is absent. For the settings a model
    /// only writes down when it departs from the usual.
    pub fn get_or<T: FromStr>(&self, key: &str, default: T) -> Result<T> {
        if self.has(key) {
            self.get(key)
        } else {
            Ok(default)
        }
    }

    /// The value of `key` as a flag. Accepts the spellings the C++ reader accepts.
    pub fn get_bool(&self, key: &str) -> Result<bool> {
        let value = self.get_str(key)?;
        match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(Error::format(format!(
                "{}: key {key:?} holds {other:?}, which is not a boolean",
                self.name
            ))),
        }
    }

    /// [`IniSection::get_bool`] with a default for an absent key.
    pub fn get_bool_or(&self, key: &str, default: bool) -> Result<bool> {
        if self.has(key) {
            self.get_bool(key)
        } else {
            Ok(default)
        }
    }
}

/// A parsed configuration file.
#[derive(Clone, Debug, Default)]
pub struct IniConfig {
    sections: BTreeMap<String, IniSection>,
}

impl IniConfig {
    /// Parse the text of a configuration file.
    pub fn parse(text: &str) -> Result<IniConfig> {
        let mut sections: BTreeMap<String, IniSection> = BTreeMap::new();
        let mut current: Option<String> = None;

        for (number, line) in text.lines().enumerate() {
            let line = strip_comment(line).trim();
            if line.is_empty() {
                continue;
            }

            if let Some(rest) = line.strip_prefix('[') {
                let name = rest.strip_suffix(']').ok_or_else(|| {
                    Error::format(format!("line {}: unterminated section header", number + 1))
                })?;
                let name = name.trim().to_string();
                sections.entry(name.clone()).or_insert_with(|| IniSection {
                    name: name.clone(),
                    entries: BTreeMap::new(),
                });
                current = Some(name);
                continue;
            }

            let (key, value) = line.split_once('=').ok_or_else(|| {
                Error::format(format!(
                    "line {}: {line:?} is not a key = value",
                    number + 1
                ))
            })?;
            let section = current.as_ref().ok_or_else(|| {
                Error::format(format!(
                    "line {}: {key:?} is outside any section",
                    number + 1
                ))
            })?;

            sections
                .get_mut(section)
                .expect("the current section was inserted when its header was read")
                .entries
                .insert(key.trim().to_string(), value.trim().to_string());
        }

        Ok(IniConfig { sections })
    }

    /// The section called `name`.
    pub fn section(&self, name: &str) -> Result<&IniSection> {
        self.sections
            .get(name)
            .ok_or_else(|| Error::format(format!("section [{name}] not found")))
    }

    /// Whether the file has a section called `name`.
    pub fn has_section(&self, name: &str) -> bool {
        self.sections.contains_key(name)
    }
}

/// Drops a trailing `;` or `#` comment, which may be the whole line.
fn strip_comment(line: &str) -> &str {
    match line.find([';', '#']) {
        Some(index) => &line[..index],
        None => line,
    }
}
