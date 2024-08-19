#!/bin/bash

set -e

if [ ! -d ffmpeg-7.0.2 ]; then
  wget -nv https://www.ffmpeg.org/releases/ffmpeg-7.0.2.tar.gz
  tar xzf ffmpeg-7.0.2.tar.gz

  rm ffmpeg-7.0.2.tar.gz
fi

cd ffmpeg-7.0.2
echo "configuring ffmpeg ..."
./configure \
    --disable-alsa \
    --disable-x86asm \
    --disable-shared \
    --enable-static \
    --enable-pic \
    --disable-doc \
    --disable-debug \
    --disable-avdevice \
    --disable-swscale \
    --disable-programs \
    --enable-ffmpeg \
    --enable-ffprobe \
    --disable-network \
    --disable-zlib \
    --disable-lzma \
    --disable-bzlib \
    --disable-iconv \
    --disable-libxcb \
    --disable-bsfs \
    --disable-indevs \
    --disable-outdevs \
    --disable-hwaccels \
    --disable-nvenc \
    --disable-videotoolbox \
    --disable-audiotoolbox \
    --disable-filters \
    --enable-filter=aresample \
    --disable-muxers \
    --enable-muxer=pcm_s16le \
    --disable-encoders \
    --enable-encoder=pcm_s16le \
    --disable-demuxers \
    --enable-demuxer=mov \
    --disable-decoders \
    --enable-decoder=aac \
    --enable-decoder=mp3 \
    --disable-parsers

make -j
