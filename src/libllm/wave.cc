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

#include "libllm/functional.h"
#include "lutil/error.h"

namespace libllm {

Wave::Wave()
    : _maxHistoryInBytes(0) {
}

Wave::Wave(std::shared_ptr<WaveStream> waveStream)
    : _maxHistoryInBytes(60 * waveStream->getBytesPerSample() * waveStream->getSampleRate()),
      _waveStream(waveStream) {
}

Tensor Wave::toTensor(lut::Span<const Byte> data) {
  int numSamples = static_cast<int>(data.size() / 2);
  if (data.size() % 2 != 0) {
    throw lut::AbortedError("Wave: invalid size of data");
  }

  std::vector<float> wave(numSamples);
  const int16_t *phData = reinterpret_cast<const int16_t *>(data.data());
  for (int i = 0; i < numSamples; ++i) {
    wave[i] = static_cast<float>(phData[i]) / 32768.0f;
  }

  return Tensor::create({numSamples}, lut::makeConstSpan(wave));
}

void Wave::updateHistoryBuffer(lut::Span<const Byte> data) {
  _historyBuffer.insert(_historyBuffer.end(), data.begin(), data.end());
  if (_historyBuffer.size() > _maxHistoryInBytes) {
    int64_t nbToErase = _historyBuffer.size() - _maxHistoryInBytes;
    _historyBuffer.erase(_historyBuffer.begin(), _historyBuffer.begin() + nbToErase);
  }
}

int64_t Wave::readFromBuffer(lut::Span<Byte> data) {
  auto endIt = _readBuffer.size() > data.size() ? _readBuffer.begin() + data.size()
                                                : _readBuffer.end();

  std::copy(_readBuffer.begin(), endIt, data.begin());
  _readBuffer.erase(_readBuffer.begin(), endIt);

  return endIt - _readBuffer.begin();
}

Tensor Wave::read(lut::Duration duration) {
  int64_t nbTotalSize = durationToNumBytes(duration);
  std::vector<Byte> data(nbTotalSize);

  // read from buffer.
  int64_t nbRead = 0;
  if (!_readBuffer.empty()) {
    nbRead = readFromBuffer(lut::makeSpan(data));
  }

  int64_t nbToRead = nbTotalSize - nbRead;
  CHECK(nbToRead >= 0);

  // read from stream.
  if (nbToRead) {
    nbRead += _waveStream->read(lut::makeSpan(data).subspan(nbRead));
    _readOffset += numBytesToDuration(nbRead);
  }

  lut::Span<const Byte> readData = lut::makeConstSpan(data).subspan(0, nbRead);
  updateHistoryBuffer(readData);

  CHECK(_waveStream->getBytesPerSample() == 2);
  return toTensor(readData);
}

void Wave::seek(lut::Duration offset) {
  CHECK(offset <= _readOffset);
  CHECK(_readBuffer.empty()) << "unable to call seek() twice.";

  int64_t nbToBackward = durationToNumBytes(_readOffset - offset);
  CHECK(nbToBackward <= _historyBuffer.size());

  if (nbToBackward) {
    _readBuffer.insert(
        _readBuffer.begin(),
        _historyBuffer.end() - nbToBackward,
        _historyBuffer.end());

    _historyBuffer.erase(_historyBuffer.end() - nbToBackward, _historyBuffer.end());
  }
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
  return _waveStream->eof();
}

lut::Duration Wave::tell() const {
  lut::Duration bufDur = numBytesToDuration(_readBuffer.size());
  return _readOffset - bufDur;
}

}  // namespace libllm
