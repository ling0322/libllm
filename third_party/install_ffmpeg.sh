#!/bin/bash

set -e

if [ ! -d ffmpeg ]; then
  wget -nv https://www.ffmpeg.org/releases/ffmpeg-7.0.2.tar.gz
  tar xzf ffmpeg-7.0.2.tar.gz

  rm ffmpeg-7.0.2.tar.gz
  mv ffmpeg-7.0.2 ffmpeg
fi


cd ffmpeg
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
    --enable-demuxer=aac \
    --enable-demuxer=flac \
    --enable-demuxer=flv \
    --enable-demuxer=ogg \
    --enable-demuxer=s16be \
    --enable-demuxer=s16le \
    --enable-demuxer=s32be \
    --enable-demuxer=s32le \
    --enable-demuxer=matroska \
    --enable-demuxer=mov \
    --enable-demuxer=mp3 \
    --enable-demuxer=wav \
    --enable-demuxer=xwma \
    --disable-decoders \
    --enable-decoder=aac \
    --enable-decoder=ac3 \
    --enable-decoder=flac \
    --enable-decoder=mp2 \
    --enable-decoder=mp3 \
    --enable-decoder=opus \
    --enable-decoder=pcm_alaw \
    --enable-decoder=pcm_f32be \
    --enable-decoder=pcm_f32le \
    --enable-decoder=pcm_mulaw \
    --enable-decoder=pcm_s16be \
    --enable-decoder=pcm_s16le \
    --enable-decoder=vorbis \
    --enable-decoder=wmav1 \
    --enable-decoder=wmav2 \
    --disable-parsers

make -j
