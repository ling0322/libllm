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

#pragma once

#include <stdint.h>
#include <stdio.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "lut/noncopyable.h"
#include "lut/span.h"

namespace lut {

// -----------------------------------------------------------------------------------------------+
// class Reader                                                                                   |
// -----------------------------------------------------------------------------------------------+

class Reader {
 public:
  static constexpr int kDefaultBufferSize = 4096;

  // create the BufferedReader instance by Reader and the buffer size.
  Reader(int buffer_size = kDefaultBufferSize);

  // read `buffer.size()` bytes and fill the `buffer`. Throw if EOF reached or other errors occured.
  void readSpan(Span<int8_t> buffer);

  // read a variable from reader
  template<typename T>
  T readValue();

  // read a string of `n` bytes from file.
  std::string readString(int n);

  // Read a line from the stream. If EOF is reached, return an empty string. Throw an error if any
  // other error occurs.
  std::string readLine();

 protected:
  // Implemented by sub classes. Read a span from file and return number of bytes read. On EOF
  // reached, returns 0. Throw exceptions if other errors occured.
  virtual int64_t readData(Span<int8_t> buffer) = 0;

 private:
  FixedArray<int8_t> _buffer;

  // write and read position in buffer
  int64_t _w;
  int64_t _r;

  // read at most `dest.size()` bytes from buffer and return the number of bytes read. The number
  // will less than `dest.size()` once no enough bytes in buffer.
  int64_t readFromBuffer(Span<int8_t> dest);

  // read next buffer from Reader. Return the number of bytes read.
  int64_t readNextBuffer();
};

template<typename T>
T Reader::readValue() {
  T value;
  readSpan(lut::makeSpan(reinterpret_cast<int8_t *>(&value), sizeof(T)));

  return value;
}

// -----------------------------------------------------------------------------------------------+
// class ReadableFile                                                                             |
// -----------------------------------------------------------------------------------------------+

// reader for local file.
class ReadableFile : public Reader {
 public:
  static std::unique_ptr<ReadableFile> open(const std::string &filename);

  ~ReadableFile();
  ReadableFile(ReadableFile &) = delete;
  ReadableFile &operator=(ReadableFile &) = delete;

 protected:
  // implements interface Reader
  int64_t readData(Span<int8_t> buffer) override;

 private:
  FILE *_fp;

  ReadableFile();
};

}  // namespace lut
