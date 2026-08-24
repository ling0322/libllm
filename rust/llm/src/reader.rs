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

//! Reading the little-endian binary records that the model formats are written in.

use std::io::Read;

use crate::error::{Error, Result};

/// The primitive reads every format in this crate is built from. Implemented for anything that
/// reads bytes, so a model file and a buffer in a test go through the same code.
pub trait BinaryRead: Read {
    fn read_exact_bytes(&mut self, count: usize) -> Result<Vec<u8>> {
        let mut buffer = vec![0u8; count];
        self.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut bytes = [0u8; 2];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut bytes = [0u8; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut bytes = [0u8; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut bytes = [0u8; 1];
        self.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }

    /// Reads `count` bytes as text. The formats here write tags and names as raw bytes, so this
    /// keeps them as they were written rather than assuming they are valid UTF-8.
    fn read_string(&mut self, count: usize) -> Result<String> {
        let bytes = self.read_exact_bytes(count)?;
        String::from_utf8(bytes).map_err(|_| Error::format("string is not valid UTF-8"))
    }

    /// Reads a tag and checks it, which is how each format announces what comes next.
    fn expect_tag(&mut self, expected: &str) -> Result<()> {
        let tag = self.read_string(expected.len())?;
        if tag != expected {
            return Err(Error::format(format!(
                "expected tag {expected:?}, got {tag:?}"
            )));
        }
        Ok(())
    }
}

impl<R: Read> BinaryRead for R {}
