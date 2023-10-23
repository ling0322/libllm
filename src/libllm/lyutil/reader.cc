// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "lyutil/reader.h"

#include "lyutil/error.h"
#include "lyutil/log.h"
#include "lyutil/strings.h"

namespace ly {

std::vector<int8_t> readFile(const std::string &filename) {
  std::vector<int8_t> data;
  std::vector<int8_t> chunk(4096);

  auto fp = ReadableFile::open(filename);

  for (; ; ) {
    int cbytes = fp->read(makeSpan(chunk));
    if (cbytes) {
      data.insert(data.end(), chunk.begin(), chunk.begin() + cbytes);
    } else {
      break;
    }
  }

  return data;
}

// -- class BufferedReader -------------------------------------------------------------------------

BufferedReader::BufferedReader(int buffer_size)
    : _buffer(buffer_size), 
      _w(0),
      _r(0) {}

int BufferedReader::readFromBuffer(Span<int8_t> dest) {
  int n = std::min(static_cast<int>(dest.size()), _w - _r);

  int8_t *begin = _buffer.begin() + _r;
  std::copy(begin, begin + n, dest.begin());

  _r += n;
  return n;
}

int BufferedReader::readNextBuffer() {
  CHECK(_w - _r == 0);
  _r = 0;
  _w = 0;
  _w = read(makeSpan(_buffer));

  return _w;
}

void BufferedReader::readSpan(Span<int8_t> span) {
  Span<int8_t>::iterator it = span.begin();

  int bytesRead = readFromBuffer(span);
  
  while (bytesRead < span.size()) {
    int n = readNextBuffer();
    if (!n) {
      throw OutOfRangeError("unexcpected EOF");
    }

    bytesRead += readFromBuffer(span.subspan(bytesRead));
  }
}

std::string BufferedReader::readString(int n) {
  CHECK(n > 0);

  std::vector<int8_t> buffer(n);
  readSpan(makeSpan(buffer));

  return std::string(buffer.begin(), buffer.end());
}

// -- class ReadableFile ---------------------------------------------------------------------------

ReadableFile::ReadableFile() : _fp(nullptr) {}
ReadableFile::~ReadableFile() {
  if (_fp) {
    fclose(_fp);
    _fp = nullptr;
  }
}

std::unique_ptr<ReadableFile> ReadableFile::open(const std::string &filename) {
  std::unique_ptr<ReadableFile> fp{new ReadableFile()};
  fp->_fp = fopen(filename.c_str(), "rb");
  if (fp->_fp == nullptr) {
    throw AbortedError(ly::sprintf("failed to open file %s.", filename));
  }

  return fp;
}

int ReadableFile::read(Span<int8_t> buffer) {
  CHECK(buffer.size() != 0);
  int n = fread(buffer.data(), sizeof(int8_t), buffer.size(), _fp);

  if ((!n) && !feof(_fp)) {
    throw AbortedError("failed to read file.");
  }

  return n;
}

// -- class Scanner --------------------------------------------------------------------------------

Scanner::Scanner(Reader *reader) : _reader(reader), _buffer(BufferSize) {}

bool Scanner::scan() {
  _text.clear();

  for (; ; ) {
    if (_bufferSpan.empty()) {
      bool ok = readBuffer();
      if (!ok) {
        return !_text.empty();
      }
    }

    auto it = _bufferSpan.begin();
    for (; it < _bufferSpan.end() && *it != '\n'; ++it) {
    }
    if (it != _bufferSpan.end()) {
      _text.insert(_text.end(), _bufferSpan.begin(), it);
      _bufferSpan = _bufferSpan.subspan(it - _bufferSpan.begin() + 1);
      return true;
    } else {
      _text.insert(_text.end(), _bufferSpan.begin(), _bufferSpan.end());
      _bufferSpan = Span<int8_t>();
    }
  }

  // will not reach here.
  NOT_IMPL();
  return false;
}

const std::string &Scanner::getText() const {
  return _text;
}

bool Scanner::readBuffer() {
  int n = _reader->read(makeSpan(_buffer));
  if (!n) {
    return false;
  }

  _bufferSpan = makeSpan(_buffer.data(), n);
  return true;
}

} // namespace ly
