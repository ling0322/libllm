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

#include "libllm/generator.h"
#include "libllm/wave.h"
#include "libllm/whisper.h"
#include "lutil/time.h"

namespace libllm {
namespace whisper {

struct RecognitionResult {
  lut::Duration begin;
  lut::Duration end;
  std::string language;
  std::string text;
};

/// @brief decoder for 30s chunks with greedy search.
class WhisperChunkGreedySearchDecoder {
 public:
  static std::shared_ptr<WhisperChunkGreedySearchDecoder> fromModel(
      std::shared_ptr<WhisperModel> model);
  ~WhisperChunkGreedySearchDecoder() = default;

  void setLang(const std::string &langCode);
  void setTemperature(float temperature);

  /// @brief decode one token.
  /// @return the whole decode result.
  std::vector<RecognitionResult> decode(Tensor wave);

 private:
  StateMap _kvCache;
  float _temperature;

  int _noSpeechToken;
  int _targetLangToken;
  int _langTokenEn;
  int _langTokenSu;

  int _lastTimeToken;
  int _timestamp0000;
  int _timestamp3000;
  int _eotToken;
  int _transcribeToken;
  int _translateToken;
  int _noTimestampToken;
  int _startOfTranscriptToken;

  int _lastTimeTokenIdx;
  float _noSpeechProb;
  lut::Duration _audioLength;

  std::vector<LongType> _history;
  std::shared_ptr<WhisperModel> _model;

  void processLogits(Tensor logits);
  std::string parseLangToken(int tokenId) const;
  lut::Duration parseTimestampToken(int tokenId) const;
  bool isTimestampToken(int tokenId) const;
  void updateHistory(int tokenId);

  Tensor applySoftmax(Tensor logits);
  void inferLang();

  /// @brief decode one token from input audio.
  /// @return the next token.
  int decodeToken();

  WhisperChunkGreedySearchDecoder();
};

class WhisperDecoder {
 public:
  static std::shared_ptr<WhisperDecoder> create(
      std::shared_ptr<WhisperModel> model,
      std::shared_ptr<Wave> wave);

  /// @brief get next result from decoder.
  /// @return next result. nullopt if decoding is ended.
  std::optional<RecognitionResult> nextResult();

 private:
  std::shared_ptr<WhisperChunkGreedySearchDecoder> _chunkDecoder;
  std::shared_ptr<Wave> _wave;
  std::deque<RecognitionResult> _results;
  lut::Duration _waveOffset;
  bool _waveEof;

  WhisperDecoder();
};

}  // namespace whisper
}  // namespace libllm