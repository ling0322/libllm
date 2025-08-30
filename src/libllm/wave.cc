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

#include "libllm/wave.h"

#include "lutil/error.h"
#include "lynn/functional.h"

namespace libllm {

Wave::Wave()
    : _bufferOffset(0),
      _readOffset(0),
      _eof(false) {
}

Wave::Wave(std::shared_ptr<WaveStream> waveStream)
    : _waveStream(waveStream),
      _bufferOffset(0),
      _readOffset(0),
      _eof(false) {
}

ly::Tensor Wave::toTensor(lut::Span<const ly::Byte> data) {
  int numSamples = static_cast<int>(data.size() / 2);
  if (data.size() % 2 != 0) {
    throw lut::AbortedError("Wave: invalid size of data");
  }

  std::vector<float> wave(numSamples);
  const int16_t *phData = reinterpret_cast<const int16_t *>(data.data());
  for (int i = 0; i < numSamples; ++i) {
    wave[i] = static_cast<float>(phData[i]) / 32768.0f;
  }

  return ly::Tensor::create({numSamples}, lut::makeConstSpan(wave));
}

void Wave::readBlock() {
  std::vector<ly::Byte> data(BlockSize);
  int64_t nb = _waveStream->read(lut::makeSpan(data));
  _buffer.insert(_buffer.end(), data.begin(), data.begin() + nb);
}

ly::Tensor Wave::read(lut::Duration duration) {
  CHECK(_waveStream->getBytesPerSample() == 2);

  int64_t nbTotalSize = durationToNumBytes(duration);
  while (_readOffset + nbTotalSize >= _bufferOffset + _buffer.size() && !_waveStream->eof()) {
    readBlock();
  }

  std::vector<ly::Byte> data;
  CHECK(_readOffset >= _bufferOffset && _readOffset < _bufferOffset + _buffer.size());
  auto it = _buffer.begin() + (_readOffset - _bufferOffset);
  data.insert(data.end(), it, std::min(it + nbTotalSize, _buffer.end()));
  if (data.size() < nbTotalSize) {
    _eof = true;
  }

  _readOffset += data.size();
  pruneBuffer();

  if (!data.empty()) {
    return toTensor(data);
  } else {
    return ly::Tensor();
  }
}

void Wave::pruneBuffer() {
  CHECK(_readOffset >= _bufferOffset);
  if (_readOffset - _bufferOffset > BlockSize + MaxHistoryInBytes) {
    CHECK(_buffer.size() > BlockSize);
    _buffer.erase(_buffer.begin(), _buffer.begin() + BlockSize);
    _bufferOffset += BlockSize;
  }
}

void Wave::seek(lut::Duration offset) {
  int64_t offsetInBytes = durationToNumBytes(offset);
  CHECK(offsetInBytes <= _readOffset && offsetInBytes >= _bufferOffset);
  _readOffset = offsetInBytes;
}

int64_t Wave::durationToNumBytes(lut::Duration dur) const {
  int64_t nSamples = dur.totalNanoseconds() * _waveStream->getSampleRate() / 1000000000;
  return nSamples * _waveStream->getBytesPerSample();
}

lut::Duration Wave::numBytesToDuration(int64_t numBytes) const {
  int64_t numBytesPerSecond = _waveStream->getSampleRate() * _waveStream->getBytesPerSample();
  return lut::Duration::nanoseconds(numBytes * 1000000000 / numBytesPerSecond);
}

bool Wave::eof() {
  return _eof;
}

lut::Duration Wave::tell() const {
  return numBytesToDuration(_readOffset);
}

}  // namespace libllm
