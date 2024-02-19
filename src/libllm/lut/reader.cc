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

#include "libllm/lut/reader.h"

#include "libllm/lut/error.h"
#include "libllm/lut/log.h"
#include "libllm/lut/strings.h"

namespace lut {

// -- class Reader -------------------------------------------------------------------------

Reader::Reader(int buffer_size)
    : _buffer(buffer_size), 
      _w(0),
      _r(0) {}

int64_t Reader::readFromBuffer(Span<int8_t> dest) {
  int64_t n = std::min(static_cast<int64_t>(dest.size()), _w - _r);

  int8_t *begin = _buffer.begin() + _r;
  std::copy(begin, begin + n, dest.begin());

  _r += n;
  return n;
}

int64_t Reader::readNextBuffer() {
  CHECK(_w - _r == 0);
  _r = 0;
  _w = 0;
  _w = readData(makeSpan(_buffer));

  return _w;
}

void Reader::readSpan(Span<int8_t> span) {
  Span<int8_t>::iterator it = span.begin();

  int64_t bytesRead = readFromBuffer(span);
  
  while (bytesRead < static_cast<int64_t>(span.size())) {
    int64_t n = readNextBuffer();
    if (!n) {
      throw OutOfRangeError("unexcpected EOF");
    }

    bytesRead += readFromBuffer(span.subspan(bytesRead));
  }
}

std::string Reader::readString(int n) {
  CHECK(n > 0);
  if (n == 0) return "";

  std::vector<int8_t> buffer(n);
  readSpan(makeSpan(buffer));

  return std::string(buffer.begin(), buffer.end());
}

std::string Reader::readLine() {
  std::string line;

  int8_t ch;
  for (; ; ) {
    if (_w - _r == 0) {
      int64_t n = readNextBuffer();

      // on EOF.
      if (!n) return line;
    }

    ch = _buffer[_r];
    ++_r;
    line.push_back(ch);
    if (ch == '\n') return line;
  }

  NOT_IMPL();
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
    THROW(Aborted, lut::sprintf("failed to open file %s", filename));
  }

  return fp;
}

int64_t ReadableFile::readData(Span<int8_t> buffer) {
  CHECK(buffer.size() != 0);
  size_t n = fread(buffer.data(), sizeof(int8_t), buffer.size(), _fp);

  if ((!n) && !feof(_fp)) {
    throw AbortedError("failed to read file.");
  }

  return n;
}

} // namespace lut
