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

#include "libllm/wave_stream.h"

#include <memory>
#include <mutex>

#include "libllm/read_audio_ffmpeg.h"
#include "lutil/error.h"
#include "lutil/span.h"
#include "lynn/tensor.h"

namespace libllm {

std::once_flag gFFmpegPluginLoadOnce;
lut::SharedLibrary *gFFmpegLibrary = nullptr;
decltype(&llm_ffmpeg_get_err) p_llm_ffmpeg_get_err = nullptr;
decltype(&llm_ffmpeg_audio_open) p_llm_ffmpeg_audio_open = nullptr;
decltype(&llm_ffmpeg_audio_close) p_llm_ffmpeg_audio_close = nullptr;
decltype(&llm_ffmpeg_audio_read) p_llm_ffmpeg_audio_read = nullptr;

void initFFmpegPluginOnce() {
  std::call_once(gFFmpegPluginLoadOnce, []() {
    if (gFFmpegLibrary) {
      LOG(WARN) << "gFFmpegLibrary is not empty when calling loadFFmpegPluginOnce()";
      delete gFFmpegLibrary;
      gFFmpegLibrary = nullptr;
    }

    try {
      std::unique_ptr<lut::SharedLibrary> lib = lut::SharedLibrary::open("llmpluginffmpeg");
      gFFmpegLibrary = lib.release();

      p_llm_ffmpeg_get_err = gFFmpegLibrary->getFunc<decltype(llm_ffmpeg_get_err)>(
          "llm_ffmpeg_get_err");
      p_llm_ffmpeg_audio_open = gFFmpegLibrary->getFunc<decltype(llm_ffmpeg_audio_open)>(
          "llm_ffmpeg_audio_open");
      p_llm_ffmpeg_audio_close = gFFmpegLibrary->getFunc<decltype(llm_ffmpeg_audio_close)>(
          "llm_ffmpeg_audio_close");
      p_llm_ffmpeg_audio_read = gFFmpegLibrary->getFunc<decltype(llm_ffmpeg_audio_read)>(
          "llm_ffmpeg_audio_read");

    } catch (lut::Error &e) {
      p_llm_ffmpeg_get_err = nullptr;
      p_llm_ffmpeg_audio_open = nullptr;
      p_llm_ffmpeg_audio_close = nullptr;
      p_llm_ffmpeg_audio_read = nullptr;

      delete gFFmpegLibrary;
      gFFmpegLibrary = nullptr;

      LOG(ERROR) << "load FFmpeg plugin failed: " << e.what();
    }
  });
}

FFmpegWaveStream::FFmpegWaveStream()
    : _eof(false) {
}

std::shared_ptr<FFmpegWaveStream> FFmpegWaveStream::open(const std::string &filename) {
  std::shared_ptr<FFmpegWaveStream> stream{new FFmpegWaveStream()};

  initFFmpegPluginOnce();
  if (!gFFmpegLibrary) {
    throw lut::AbortedError("ffmpeg plugin not initialized.");
  }

  llm_ffmpeg_audio_reader_t *reader = p_llm_ffmpeg_audio_open(filename.c_str());
  if (!reader) {
    std::string errmsg = "unable to open file \"" + filename + "\" with ffmpeg plugin: ";
    errmsg += p_llm_ffmpeg_get_err();
    throw lut::AbortedError(errmsg);
  }

  stream->_reader = {reader, p_llm_ffmpeg_audio_close};
  return stream;
}

int FFmpegWaveStream::read(lut::Span<ly::Byte> buffer) {
  int nb = p_llm_ffmpeg_audio_read(
      _reader.get(),
      reinterpret_cast<char *>(buffer.data()),
      buffer.size());
  if (nb < 0) {
    throw lut::AbortedError(p_llm_ffmpeg_get_err());
  } else if (nb < buffer.size()) {
    _eof = true;
  }

  return nb;
}

int FFmpegWaveStream::getSampleRate() const {
  return 16000;
}

int FFmpegWaveStream::getBytesPerSample() const {
  return 2;
}

bool FFmpegWaveStream::eof() const {
  return _eof;
}

FFmpegWaveStream::~FFmpegWaveStream() {
}

}  // namespace libllm
