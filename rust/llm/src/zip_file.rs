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

//! Reading a model package, which is a zip file whose entries are all stored uncompressed.
//!
//! The archive is walked from the front, one local file header at a time, the same way the C++
//! reader does it: an entry is found by its header rather than by the central directory, and an
//! entry that was compressed is refused rather than inflated.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};
use crate::reader::BinaryRead;

/// How much of an entry to read at a time, which matters for the multi-gigabyte ones.
const READ_BUFFER: usize = 1 << 20;

const LOCAL_FILE_HEADER: u32 = 0x0403_4b50;
const CENTRAL_DIRECTORY_HEADER: u32 = 0x0201_4b50;
const END_OF_CENTRAL_DIRECTORY: u32 = 0x0605_4b50;
const ZIP64_END_OF_CENTRAL_DIRECTORY: u32 = 0x0606_4b50;
const ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR: u32 = 0x0706_4b50;

/// Where one entry's bytes sit in the archive.
#[derive(Clone, Copy, Debug)]
struct Entry {
    offset: u64,
    size: u64,
}

/// A model package, opened for reading.
pub struct ZipFile {
    file: File,
    entries: BTreeMap<String, Entry>,
}

impl ZipFile {
    /// Open the package at `path` and index what it holds.
    pub fn open(path: impl AsRef<Path>) -> Result<ZipFile> {
        let mut file = File::open(path)?;
        let entries = index(&mut file)?;
        Ok(ZipFile { file, entries })
    }

    /// The names of every entry, in order.
    pub fn names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// Whether the package holds an entry called `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Read one entry as it goes.
    ///
    /// The parameters of a model are the better part of the package and are turned into tensors a
    /// record at a time, so they are streamed rather than held in memory twice.
    pub fn open_entry(&self, name: &str) -> Result<EntryReader> {
        let entry = *self
            .entries
            .get(name)
            .ok_or_else(|| Error::format(format!("{name:?} is not in the package")))?;

        let mut file = self.file.try_clone()?;
        file.seek(SeekFrom::Start(entry.offset))?;
        Ok(EntryReader {
            file: BufReader::with_capacity(READ_BUFFER, file),
            remaining: entry.size,
        })
    }

    /// Read the whole of one entry, for the small ones that are parsed as a unit.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.open_entry(name)?.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Read the whole of one entry as text.
    pub fn read_to_string(&self, name: &str) -> Result<String> {
        let bytes = self.read(name)?;
        String::from_utf8(bytes).map_err(|_| Error::format(format!("{name:?} is not valid UTF-8")))
    }
}

/// Reads one entry's bytes, and stops at the end of it rather than running on into the next.
pub struct EntryReader {
    file: BufReader<File>,
    remaining: u64,
}

impl Read for EntryReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let wanted = buffer.len().min(self.remaining as usize);
        let read = self.file.read(&mut buffer[..wanted])?;
        self.remaining -= read as u64;
        Ok(read)
    }
}

/// Walks the archive and records where each entry's bytes begin.
fn index(file: &mut File) -> Result<BTreeMap<String, Entry>> {
    let end = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(0))?;

    let mut entries = BTreeMap::new();
    while file.stream_position()? + 4 <= end {
        let signature = file.read_u32()?;
        match signature {
            LOCAL_FILE_HEADER => {
                let (name, entry) = read_local_file(file)?;
                file.seek(SeekFrom::Start(entry.offset + entry.size))?;
                entries.insert(name, entry);
            }
            CENTRAL_DIRECTORY_HEADER => {
                // version_made .. lfh_offset, of which only the three trailing lengths matter.
                let mut header = [0u8; 42];
                file.read_exact(&mut header)?;
                let name_length = u16::from_le_bytes([header[24], header[25]]) as i64;
                let extra_length = u16::from_le_bytes([header[26], header[27]]) as i64;
                let comment_length = u16::from_le_bytes([header[28], header[29]]) as i64;
                file.seek(SeekFrom::Current(
                    name_length + extra_length + comment_length,
                ))?;
            }
            END_OF_CENTRAL_DIRECTORY => {
                let mut record = [0u8; 16];
                file.read_exact(&mut record)?;
                let comment_length = u16::from_le_bytes([record[14], record[15]]) as i64;
                file.seek(SeekFrom::Current(comment_length))?;
            }
            ZIP64_END_OF_CENTRAL_DIRECTORY => {
                // The recorded size counts everything after the size field itself.
                let size = file.read_u64()?;
                file.seek(SeekFrom::Current(size as i64))?;
            }
            ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR => {
                file.seek(SeekFrom::Current(16))?;
            }
            other => {
                return Err(Error::format(format!(
                    "unsupported zip signature {other:#x}"
                )));
            }
        }
    }

    Ok(entries)
}

/// Reads one local file header, leaving the file positioned at the entry's first byte.
fn read_local_file(file: &mut File) -> Result<(String, Entry)> {
    let _version = file.read_u16()?;
    let flag = file.read_u16()?;
    let compression = file.read_u16()?;
    let _last_modify_time = file.read_u16()?;
    let _last_modify_date = file.read_u16()?;
    let _crc32 = file.read_u32()?;
    let compressed_size = file.read_u32()?;
    let uncompressed_size = file.read_u32()?;
    let name_length = file.read_u16()? as usize;
    let extra_length = file.read_u16()? as u64;

    if flag & 0x08 != 0 {
        return Err(Error::format(
            "zip file contains a data descriptor, which is not supported",
        ));
    }
    if compression != 0 || compressed_size != uncompressed_size {
        return Err(Error::format("zip file is not stored uncompressed"));
    }

    let name = file.read_string(name_length)?;

    // A size of 0xffffffff means the real one is in a zip64 extra field.
    let size = if compressed_size == u32::MAX && uncompressed_size == u32::MAX {
        read_zip64_size(file, extra_length)?
    } else {
        file.seek(SeekFrom::Current(extra_length as i64))?;
        compressed_size as u64
    };

    let offset = file.stream_position()?;
    Ok((name, Entry { offset, size }))
}

/// Finds the zip64 extra field among the extra fields and takes the entry's real size from it,
/// leaving the file positioned after the whole extra field area.
fn read_zip64_size(file: &mut File, extra_length: u64) -> Result<u64> {
    let extra_start = file.stream_position()?;
    let extra_end = extra_start + extra_length;

    let mut size = None;
    while file.stream_position()? + 4 <= extra_end {
        let id = file.read_u16()?;
        let field_length = file.read_u16()? as i64;
        if id == 0x0001 {
            let _uncompressed_size = file.read_u64()?;
            size = Some(file.read_u64()?);
            break;
        }
        file.seek(SeekFrom::Current(field_length))?;
    }

    file.seek(SeekFrom::Start(extra_end))?;
    size.ok_or_else(|| Error::format("zip64 entry has no zip64 extra field"))
}
