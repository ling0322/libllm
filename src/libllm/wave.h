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

#include <deque>
#include <memory>

#include "libllm/tensor.h"
#include "libllm/wave_stream.h"
#include "lutil/span.h"
#include "lutil/time.h"

namespace libllm {

enum class WaveFormat {
  Wave16kHz16bitMonoPCM,
  Unknown,
};

/// @brief Wrap the WaveStream into a file-like Wave object. It supports 2 operations seek() and
/// read(). seek() only supports moving backward up to 1min.
class Wave {
 public:
  static constexpr int MaxHistoryInBytes = 60 * 16000 * 2;
  static constexpr int BlockSize = 5 * MaxHistoryInBytes;

  Wave();
  Wave(std::shared_ptr<WaveStream> wave_stream);

  /// @brief convert S16LE format audio date to tensor (samples).
  static Tensor toTensor(lut::Span<const Byte> data);

  Tensor read(lut::Duration length);
  void seek(lut::Duration offset);
  lut::Duration tell() const;
  bool eof();

 private:
  int64_t _bufferOffset;
  int64_t _readOffset;
  bool _eof;
  std::deque<Byte> _buffer;
  std::shared_ptr<WaveStream> _waveStream;

  int64_t durationToNumBytes(lut::Duration dur) const;
  lut::Duration numBytesToDuration(int64_t numBytes) const;

  void readBlock();
  void pruneBuffer();
};

}  // namespace libllm
