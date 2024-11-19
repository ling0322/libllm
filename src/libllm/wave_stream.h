// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include <memory>

#include "libllm/read_audio_ffmpeg.h"
#include "libllm/tensor.h"
#include "lutil/c_ptr.h"
#include "lutil/shared_library.h"
#include "lutil/span.h"

namespace libllm {

class WaveStream {
 public:
  virtual ~WaveStream() = default;

  /// @brief read buffer.size() bytes into buffer. Returns the actual bytes read. On EOF, returns 0.
  /// On other errors, throw an exception.
  /// @param buffer the dest buffer.
  /// @return bytes read.
  virtual int read(lut::Span<Byte> buffer) = 0;

  virtual bool eof() const = 0;
  virtual int getSampleRate() const = 0;
  virtual int getBytesPerSample() const = 0;
};

class FFmpegWaveStream : public WaveStream {
 public:
  static std::shared_ptr<FFmpegWaveStream> open(const std::string &filename);
  ~FFmpegWaveStream();

  int read(lut::Span<Byte> buffer) override;
  int getSampleRate() const override;
  int getBytesPerSample() const override;
  bool eof() const override;

 private:
  lut::c_ptr<llm_ffmpeg_audio_reader_t> _reader;
  bool _eof;

  FFmpegWaveStream();
};

}  // namespace libllm
