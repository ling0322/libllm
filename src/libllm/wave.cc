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

Tensor Wave::read(lut::Span<const Byte> data, WaveFormat format) {
  if (format == WaveFormat::Wave16kHz16bitMonoPCM) {
    int numSamples = static_cast<int>(data.size() / 2);
    if (data.size() % 2 != 0) {
      throw lut::AbortedError("Wave::read: invalid size of data");
    }

    std::vector<float> wave(numSamples);
    const int16_t *phData = reinterpret_cast<const int16_t *>(data.data());
    for (int i = 0; i < numSamples; ++i) {
      wave[i] = static_cast<float>(phData[i]) / 32768.0f;
    }

    return Tensor::create({numSamples}, lut::makeConstSpan(wave));
  } else {
    NOT_IMPL();
  }
}

}  // namespace libllm
