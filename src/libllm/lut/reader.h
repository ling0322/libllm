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

#include <stdio.h>
#include <stdint.h>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "libllm/lut/noncopyable.h"
#include "libllm/lut/span.h"

namespace lut {

// read all bytes from `filename`.
std::vector<int8_t> readFile(const std::string &filename);

// -- class Reader ---------------------------------------------------------------------------------

// base class for all readers
class Reader {
 public:
  virtual ~Reader() = default;
  
  // Read a buffer from file and return number of bytes read. On EOF reached, returns 0. Throw
  // exceptio if other errors occured.
  // This is a low level file read function, please use other high level functions instead.
  virtual int64_t read(Span<int8_t> buffer) = 0;
};

// -- class BufferedReader -------------------------------------------------------------------------

// implements buffered read.
class BufferedReader : public Reader {
 public:
  static constexpr int kDefaultBufferSize = 4096;

  // create the BufferedReader instance by Reader and the buffer size.
  BufferedReader(int buffer_size = kDefaultBufferSize);
  
  // read `buffer.size()` bytes and fill the `buffer`. Throw if EOF reached or other errors occured.
  void readSpan(Span<int8_t> buffer);

  // read a variable from reader
  template<typename T>
  T readValue();

  // read a string of `n` bytes from file
  std::string readString(int n);

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
T BufferedReader::readValue() {
  T value;
  readSpan(lut::makeSpan(reinterpret_cast<int8_t *>(&value), sizeof(T)));
  
  return value;
}

// -- class Scanner --------------------------------------------------------------------------------

// interface to read file line by line.
// Example:
//   Scanner scanner(fp);
//   while (scanner.scan()) {
//     const std::string &s = scanner.getText(); 
//   }
class Scanner {
 public:
  static constexpr int BufferSize = 4096;

  Scanner(Reader *reader);

  // returns false once EOF reached.
  bool scan();
  const std::string &getText() const;

 private:
  Reader *_reader;
  std::string _text;

  FixedArray<int8_t> _buffer;
  Span<int8_t> _bufferSpan;

  bool readBuffer();
};


// -- class ReadableFile ---------------------------------------------------------------------------

// reader for local file.
class ReadableFile : public BufferedReader,
                     private NonCopyable {
 public:
  static std::unique_ptr<ReadableFile> open(const std::string &filename);

  ~ReadableFile();

  // implements interface Reader
  int64_t read(Span<int8_t> buffer) override;

 private:
  FILE *_fp;

  ReadableFile();
};

} // namespace lut


