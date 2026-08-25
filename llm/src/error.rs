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

//! The failures this crate reports, which are mostly about a model package that does not hold
//! what it was expected to.

use std::fmt;
use std::io;

/// What went wrong. Reading a model means reading a file someone else wrote, so most of these say
/// that the file did not match the format rather than that the caller did anything wrong.
#[derive(Debug)]
pub enum Error {
    /// The file could not be read at all.
    Io(io::Error),
    /// The file was read but does not hold what the format calls for.
    Format(String),
    /// The model package is missing something the model needs, or holds it in the wrong shape.
    Model(String),
    /// A tensor operation failed.
    Tensor(crate::flint::Error),
}

impl Error {
    pub(crate) fn format(message: impl Into<String>) -> Error {
        Error::Format(message.into())
    }

    pub(crate) fn model(message: impl Into<String>) -> Error {
        Error::Model(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(error) => write!(f, "{error}"),
            Error::Format(message) => write!(f, "{message}"),
            Error::Model(message) => write!(f, "{message}"),
            Error::Tensor(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(error) => Some(error),
            Error::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::Io(error)
    }
}

impl From<crate::flint::Error> for Error {
    fn from(error: crate::flint::Error) -> Error {
        Error::Tensor(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
