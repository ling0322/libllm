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

char *llm_ffmpeg_plugin_get_err() {
  errmsg[sizeof(errmsg) - 1] = '\0';
  return errmsg;
}

int32_t llm_ffmpeg_plugin_read_16khz_mono_pcm_from_media_file(
    const char *filename,
    char **output_buffer,
    int32_t *output_size) {
  AVFormatContext *format_ctx = nullptr;
  AVCodecContext *codec_ctx = nullptr;
  AVStream *audio_stream = nullptr;
  const AVCodec *codec = nullptr;
  struct SwrContext *swr_ctx = nullptr;
  AVPacket packet;
  AVFrame *frame = nullptr;

  int audio_stream_index = -1;
  int ret;

  if ((ret = avformat_open_input(&format_ctx, filename, nullptr, nullptr)) < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not open input file \"%s\"", filename);
    return ret;
  }

  // find stream info
  if ((ret = avformat_find_stream_info(format_ctx, nullptr)) < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not find stream information");
    return ret;
  }

  // find the audio stream
  audio_stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
  if (audio_stream_index < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not find audio stream in input file");
    return audio_stream_index;
  }
  audio_stream = format_ctx->streams[audio_stream_index];

  // get the codec context
  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    snprintf(errmsg, sizeof(errmsg), "Could not allocate codec context");
    return AVERROR(ENOMEM);
  }
  avcodec_parameters_to_context(codec_ctx, audio_stream->codecpar);

  // open the codec
  if ((ret = avcodec_open2(codec_ctx, codec, nullptr)) < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not open codec");
    return ret;
  }

  // initialize resampler
  AVChannelLayout ch_layout_output;
  av_channel_layout_default(&ch_layout_output, 1);

  ret = swr_alloc_set_opts2(
      &swr_ctx,
      &ch_layout_output,       // Output channel layout
      AV_SAMPLE_FMT_S16,       // Output sample format
      16000,                   // Output sample rate
      &codec_ctx->ch_layout,   // Input channel layout
      codec_ctx->sample_fmt,   // Input sample format
      codec_ctx->sample_rate,  // Input sample rate
      0,
      nullptr);
  if (ret < 0 || swr_init(swr_ctx) < 0) {
    snprintf(errmsg, sizeof(errmsg), "Could not allocate resample context");
    return AVERROR(ENOMEM);
  }

  // allocate memory for decoding
  frame = av_frame_alloc();
  if (!frame) {
    snprintf(errmsg, sizeof(errmsg), "Could not allocate audio frame");
    return AVERROR(ENOMEM);
  }

  // Prepare output buffer
  *output_buffer = nullptr;
  *output_size = 0;

  // Read and decode audio frames
  while (av_read_frame(format_ctx, &packet) >= 0) {
    if (packet.stream_index == audio_stream_index) {
      if (avcodec_send_packet(codec_ctx, &packet) == 0) {
        while (avcodec_receive_frame(codec_ctx, frame) == 0) {
          uint8_t *out_buf;
          int out_linesize;
          int out_samples = av_rescale_rnd(
              swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
              16000,
              codec_ctx->sample_rate,
              AV_ROUND_UP);

          // Allocate memory for resampled data
          av_samples_alloc(&out_buf, &out_linesize, 1, out_samples, AV_SAMPLE_FMT_S16, 0);

          // Resample the data
          int resampled_data = swr_convert(
              swr_ctx,
              &out_buf,
              out_samples,
              (const uint8_t **)frame->data,
              frame->nb_samples);

          // Calculate the size of the resampled data
          int data_size = av_samples_get_buffer_size(
              &out_linesize,
              1,
              resampled_data,
              AV_SAMPLE_FMT_S16,
              1);

          // Reallocate output buffer to append new data
          *output_buffer = (char *)realloc(*output_buffer, *output_size + data_size);
          memcpy(*output_buffer + *output_size, out_buf, data_size);
          *output_size += data_size;

          // Free the allocated buffer for resampled data
          av_free(out_buf);
        }
      }
      av_packet_unref(&packet);
    }
  }

_read_pcm_from_media_file_clean_up:
  av_frame_free(&frame);
  avcodec_free_context(&codec_ctx);
  avformat_close_input(&format_ctx);
  swr_free(&swr_ctx);

  return ret;
}
