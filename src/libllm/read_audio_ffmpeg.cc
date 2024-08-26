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

#include "libllm/read_audio_ffmpeg.h"

#include <algorithm>
#include <deque>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

thread_local char errmsg[256];

struct llm_ffmpeg_audio_reader_t {
  AVFrame *frame;
  AVFormatContext *formatCtx;
  AVCodecContext *codecCtx;
  SwrContext *swrCtx;
  int audioStreamIndex;
  uint8_t *resampleBuffer;
  int resampleBufferSize;
  int resampleLineSize;
  bool eof;

  std::deque<char> buffer;

  llm_ffmpeg_audio_reader_t();
  ~llm_ffmpeg_audio_reader_t();
  void ensureResampleBuffer(int nSamples);
};

llm_ffmpeg_audio_reader_t::llm_ffmpeg_audio_reader_t()
    : frame(nullptr),
      formatCtx(nullptr),
      codecCtx(nullptr),
      swrCtx(nullptr),
      audioStreamIndex(-1),
      resampleBuffer(nullptr),
      resampleBufferSize(0),
      resampleLineSize(0),
      eof(true) {
}

llm_ffmpeg_audio_reader_t::~llm_ffmpeg_audio_reader_t() {
  if (frame) av_frame_free(&frame);
  if (codecCtx) avcodec_free_context(&codecCtx);
  if (formatCtx) avformat_close_input(&formatCtx);
  if (swrCtx) swr_free(&swrCtx);
  if (resampleBuffer) av_free(resampleBuffer);

  formatCtx = nullptr;
  codecCtx = nullptr;
  swrCtx = nullptr;
  audioStreamIndex = -1;
  resampleBuffer = nullptr;
  resampleBufferSize = 0;
  resampleLineSize = 0;
  eof = true;
}

void llm_ffmpeg_audio_reader_t::ensureResampleBuffer(int nSamples) {
  if (resampleBufferSize >= nSamples) {
    return;
  }

  if (resampleBuffer) av_free(resampleBuffer);
  int ret = av_samples_alloc(&resampleBuffer, &resampleLineSize, 1, nSamples, AV_SAMPLE_FMT_S16, 0);
  if (ret < 0) {
    fprintf(stderr, "failed to alloc samples, retcode=%d\n", ret);
    abort();
  }

  resampleBufferSize = nSamples;
}

llm_ffmpeg_audio_reader_t *llm_ffmpeg_audio_open(const char *filename) {
  std::unique_ptr<llm_ffmpeg_audio_reader_t> reader = std::make_unique<llm_ffmpeg_audio_reader_t>();

  int ret = avformat_open_input(&reader->formatCtx, filename, nullptr, nullptr);
  if (ret < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not open input file \"%s\"", filename);
    return nullptr;
  }

  // find stream info
  ret = avformat_find_stream_info(reader->formatCtx, nullptr);
  if (ret < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not find stream information");
    return nullptr;
  }

  // find the audio stream
  const AVCodec *codec = nullptr;
  reader->audioStreamIndex = av_find_best_stream(
      reader->formatCtx,
      AVMEDIA_TYPE_AUDIO,
      -1,
      -1,
      &codec,
      0);
  if (reader->audioStreamIndex < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not find audio stream in input file");
    return nullptr;
  }
  AVStream *audioStream = reader->formatCtx->streams[reader->audioStreamIndex];

  // get the codec context
  reader->codecCtx = avcodec_alloc_context3(codec);
  if (!reader->codecCtx) {
    snprintf(errmsg, sizeof(errmsg), "Could not allocate codec context");
    return nullptr;
  }
  avcodec_parameters_to_context(reader->codecCtx, audioStream->codecpar);

  // open the codec
  if ((ret = avcodec_open2(reader->codecCtx, codec, nullptr)) < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not open codec");
    return nullptr;
  }

  // initialize resampler
  AVChannelLayout ch_layout_output;
  av_channel_layout_default(&ch_layout_output, 1);

  ret = swr_alloc_set_opts2(
      &reader->swrCtx,
      &ch_layout_output,              // Output channel layout
      AV_SAMPLE_FMT_S16,              // Output sample format
      16000,                          // Output sample rate
      &reader->codecCtx->ch_layout,   // Input channel layout
      reader->codecCtx->sample_fmt,   // Input sample format
      reader->codecCtx->sample_rate,  // Input sample rate
      0,
      nullptr);
  if (ret < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not allocate resample context");
    return nullptr;
  }

  ret = swr_init(reader->swrCtx);
  if (ret < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not initialize resample context");
    return nullptr;
  }

  // allocate memory for decoding
  reader->frame = av_frame_alloc();
  if (!reader->frame) {
    snprintf(errmsg, sizeof(errmsg), "Could not allocate audio frame");
    return nullptr;
  }

  reader->eof = false;
  return reader.release();
}

void llm_ffmpeg_audio_close(llm_ffmpeg_audio_reader_t *reader) {
  delete reader;
}

int32_t llm_ffmpeg_audio_read(llm_ffmpeg_audio_reader_t *reader, char *buf, int32_t buf_size) {
  while ((!reader->eof) && buf_size > reader->buffer.size()) {
    AVPacket packet;
    int ret = av_read_frame(reader->formatCtx, &packet);
    if (ret < 0) {
      reader->eof = true;
      break;
    }

    if (packet.stream_index == reader->audioStreamIndex) {
      if (avcodec_send_packet(reader->codecCtx, &packet) == 0) {
        while (avcodec_receive_frame(reader->codecCtx, reader->frame) == 0) {
          int outSamples = av_rescale_rnd(
              swr_get_delay(reader->swrCtx, reader->codecCtx->sample_rate) +
                  reader->frame->nb_samples,
              16000,
              reader->codecCtx->sample_rate,
              AV_ROUND_UP);

          // Allocate memory for resampled data
          reader->ensureResampleBuffer(outSamples);

          // Resample the data
          int ret = swr_convert(
              reader->swrCtx,
              &reader->resampleBuffer,
              outSamples,
              (const uint8_t **)reader->frame->data,
              reader->frame->nb_samples);
          if (ret < 0) {
            snprintf(errmsg, sizeof(errmsg), "Could not convert samples");
            return ret;
          }

          // Calculate the size of the resampled data
          int destBufferSize = av_samples_get_buffer_size(
              &reader->resampleLineSize,
              1,
              ret,
              AV_SAMPLE_FMT_S16,
              1);

          // Reallocate output buffer to append new data
          reader->buffer.insert(
              reader->buffer.end(),
              reader->resampleBuffer,
              reader->resampleBuffer + destBufferSize);
        }
      }
    }
  }

  int nBytesToCopy = std::min(static_cast<int>(reader->buffer.size()), buf_size);
  std::copy(reader->buffer.begin(), reader->buffer.begin() + nBytesToCopy, buf);
  reader->buffer.erase(reader->buffer.begin(), reader->buffer.begin() + nBytesToCopy);

  return nBytesToCopy;
}

const char *llm_ffmpeg_get_err() {
  return errmsg;
}
