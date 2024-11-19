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

#include "libllm/whisper_decoder.h"

#include <memory>

#include "libllm/functional.h"
#include "lutil/error.h"

namespace libllm {
namespace whisper {

// -----------------------------------------------------------------------------------------------+
// class WhisperChunkGreedySearchDecoder                                                          |
// -----------------------------------------------------------------------------------------------+

WhisperChunkGreedySearchDecoder::WhisperChunkGreedySearchDecoder()
    : _temperature(1.0),
      _noSpeechToken(-1),
      _targetLangToken(-1),
      _langTokenEn(-1),
      _langTokenSu(-1),
      _lastTimeToken(-1),
      _timestamp0000(-1),
      _timestamp3000(-1),
      _eotToken(-1),
      _transcribeToken(-1),
      _translateToken(-1),
      _noTimestampToken(-1),
      _lastTimeTokenIdx(-1),
      _startOfTranscriptToken(-1),
      _noSpeechProb(0.0) {
}

std::shared_ptr<WhisperChunkGreedySearchDecoder> WhisperChunkGreedySearchDecoder::fromModel(
    std::shared_ptr<WhisperModel> model) {
  std::shared_ptr<WhisperChunkGreedySearchDecoder> decoder{new WhisperChunkGreedySearchDecoder()};

  const Vocab *vocab = model->getVocab();
  decoder->_model = model;
  decoder->_noSpeechToken = vocab->findControlToken("<|nospeech|>");
  decoder->_langTokenEn = vocab->findControlToken("<|en|>");
  decoder->_langTokenSu = vocab->findControlToken("<|su|>");
  decoder->_timestamp0000 = vocab->findControlToken("<|0.00|>");
  decoder->_timestamp3000 = vocab->findControlToken("<|30.00|>");
  decoder->_eotToken = vocab->findControlToken("<|endoftext|>");
  decoder->_transcribeToken = vocab->findControlToken("<|transcribe|>");
  decoder->_translateToken = vocab->findControlToken("<|translate|>");
  decoder->_noTimestampToken = vocab->findControlToken("<|notimestamps|>");
  decoder->_startOfTranscriptToken = vocab->findControlToken("<|startoftranscript|>");

  return decoder;
}

Tensor WhisperChunkGreedySearchDecoder::applySoftmax(Tensor logits) {
  CHECK(logits.getDim() == 3 && logits.getShape(0) == 1 && logits.getShape(1) == 1);

  Tensor x = logits.subtensor(0).subtensor(0);
  if (_temperature != 1.0f) {
    x = F::mul(x, 1.0f / _temperature);
  }

  processLogits(x);

  x = F::softmax(x);
  if (x.getDType() == DType::kFloat16) {
    x = F::cast(x, DType::kFloat);
  }
  if (x.getDevice().getType() == Device::kCuda) {
    x = F::to(Device::kCpu, x);
  }

  return x;
}

void WhisperChunkGreedySearchDecoder::inferLang() {
  std::vector<LongType> tokenIds{_startOfTranscriptToken};
  Tensor inputs = Tensor::create<LongType>({1, static_cast<int>(tokenIds.size())}, tokenIds);
  inputs = F::to(_model->getDevice(), inputs);

  Tensor logits = _model->prefillPrompt(_kvCache, inputs);
  Tensor prob = applySoftmax(logits);

  CHECK(prob.getDim() == 1 && prob.getStride(0) == 1);
  float *probData = prob.getData<float>();

  // get no speech prob.
  _noSpeechProb = probData[_noSpeechToken];

  // predict language if not specified.
  if (_targetLangToken < 0) {
    const float *pMaxLangProb = std::max_element(
        probData + _langTokenEn,
        probData + _langTokenSu + 1);
    _targetLangToken = pMaxLangProb - probData;
  }

  _model->decode(_kvCache, _targetLangToken);
  updateHistory(_targetLangToken);
}

void WhisperChunkGreedySearchDecoder::setTranscribeMode() {
  _model->decode(_kvCache, _transcribeToken);
  updateHistory(_transcribeToken);
}

void WhisperChunkGreedySearchDecoder::processLogits(Tensor logits) {
  bool lastWasTimestamp = _history.size() >= 1 && _history.back() >= _timestamp0000;
  bool lastWasTranscribe = _history.size() >= 1 && _history.back() == _transcribeToken;
  bool penultimateWasTimestamp = _history.size() < 2 ||
                                 _history[_history.size() - 2] >= _timestamp0000 ||
                                 _history[_history.size() - 2] == _transcribeToken ||
                                 _history[_history.size() - 2] == _translateToken;

  constexpr int MaxHistory = 5;
  if (!_history.empty()) {
    Tensor history = Tensor::create<LongType>(
        {std::min<int>(_history.size(), MaxHistory)},
        lut::makeConstSpan(_history).subspan(std::max<LongType>(_history.size() - MaxHistory, 0)));
    history = F::to(logits.getDevice(), history);
    F::repetitionPenalty(logits, history, 1.5);
  }

  constexpr float Inf = std::numeric_limits<float>::infinity();
  if (lastWasTranscribe) {
    F::fill(logits.slice(-1, {_noTimestampToken, _noTimestampToken + 1}), -Inf);
  }

  if (lastWasTimestamp) {
    _lastTimeTokenIdx = static_cast<int>(_history.size());
    if (penultimateWasTimestamp) {
      F::fill(logits.slice(-1, {_timestamp0000, _timestamp3000 + 1}), -Inf);
    } else {
      F::fill(logits.slice(-1, {0, _eotToken + 1}), -Inf);
    }
  }

  if (_lastTimeToken > _timestamp0000) {
    // do not mask the <|30.00|> timestamp tag
    int endToken = std::min(_lastTimeToken + 1, _timestamp3000);
    F::fill(logits.slice(-1, {_timestamp0000, endToken}), -Inf);
  }

  Tensor probs = F::softmax(logits);
  Tensor maxText = F::max(probs.slice(-1, {0, _eotToken + 1}));
  Tensor sumTimestamp = F::sum(probs.slice(-1, {_timestamp0000, _timestamp3000 + 1}));

  maxText = F::cast(F::to(Device::getCpu(), maxText), DType::kFloat);
  sumTimestamp = F::cast(F::to(Device::getCpu(), sumTimestamp), DType::kFloat);

  float maxTextVal = *maxText.getData<float>();
  float sumTimestampVal = *sumTimestamp.getData<float>();
  if (sumTimestampVal >= maxTextVal || _history.size() - _lastTimeTokenIdx > 70) {
    F::fill(logits.slice(-1, {0, _eotToken}), -Inf);
  }
}

int WhisperChunkGreedySearchDecoder::decodeToken() {
  CHECK(!_history.empty());
  int lastToken = _history.back();

  Tensor logits = _model->decode(_kvCache, lastToken);
  Tensor probCpu = applySoftmax(logits);

  CHECK(probCpu.getDim() == 1 && probCpu.getStride(0) == 1);
  float *pProb = probCpu.getData<float>();

  const float *pMaxProb = std::max_element(pProb, pProb + probCpu.getShape(0));
  int nextToken = static_cast<int>(pMaxProb - pProb);
  if (isTimestampToken(nextToken) && nextToken != _timestamp0000) {
    --nextToken;
  }

  updateHistory(nextToken);
  return nextToken;
}

void WhisperChunkGreedySearchDecoder::updateHistory(int tokenId) {
  _history.push_back(tokenId);
  if (tokenId >= _timestamp0000 && tokenId <= _timestamp3000) {
    _lastTimeToken = tokenId;
  }
}

bool WhisperChunkGreedySearchDecoder::isTimestampToken(int tokenId) const {
  return (tokenId <= _timestamp3000) && (tokenId >= _timestamp0000);
}

lut::Duration WhisperChunkGreedySearchDecoder::parseTimestampToken(int tokenId) const {
  CHECK(isTimestampToken(tokenId));
  return lut::Duration::milliseconds((tokenId - _timestamp0000) * 20);
}

std::string WhisperChunkGreedySearchDecoder::parseLangToken(int tokenId) const {
  CHECK(tokenId >= _langTokenEn && tokenId <= _langTokenSu);
  std::string token = _model->getVocab()->getTokenString(tokenId);

  CHECK(token.size() > 4);
  return token.substr(2, token.size() - 4);
}

std::vector<RecognitionResult> WhisperChunkGreedySearchDecoder::decode(Tensor wave) {
  // reset states.
  _kvCache = StateMap();
  _history.clear();
  _lastTimeToken = -1;
  _noSpeechProb = 0.0f;
  _audioLength = lut::Duration();

  // prefill audio.
  _model->prefillAudio(_kvCache, wave);
  _audioLength = lut::Duration::nanoseconds(wave.getShape(0) * 1000000000LL / 16000);

  // prepare prompt.
  inferLang();
  setTranscribeMode();

  // generate tokens.
  std::vector<RecognitionResult> results;
  bool lastResultHasEndTimestamp = true;
  for (;;) {
    int token = decodeToken();
    if (token == _eotToken) {
      break;
    }

    lut::Duration begin = parseTimestampToken(token);
    std::string text;
    token = decodeToken();
    while (!_model->getVocab()->isControlToken(token)) {
      text += _model->getVocab()->getTokenPiece(token);
      token = decodeToken();
    }

    lut::Duration end;
    if (token == _eotToken) {
      end = _audioLength;
      lastResultHasEndTimestamp = false;
    } else {
      end = parseTimestampToken(token);
    }

    RecognitionResult result;
    result.begin = begin;
    result.end = std::min(end, _audioLength);
    result.text = text;
    result.language = parseLangToken(_targetLangToken);
    results.emplace_back(result);
    LOG(DEBUG) << result.begin.toString() << " - " << result.end.toString() << " : " << result.text;

    if (token == _eotToken) {
      break;
    }

    if (_audioLength - end <= lut::Duration::milliseconds(100)) {
      break;
    }

    // check repetition
    if (results.size() >= 3) {
      std::string_view text0 = results[results.size() - 1].text;
      std::string_view text1 = results[results.size() - 2].text;
      std::string_view text2 = results[results.size() - 3].text;
      if (text0 == text1 && text0 == text2) {
        break;
      }
    }
  }

  // remove last result if it did not have end timestamp
  if (results.size() > 1 && !lastResultHasEndTimestamp) {
    results.pop_back();
  }

  return results;
}

// -----------------------------------------------------------------------------------------------+
// class WhisperDecoder                                                                           |
// -----------------------------------------------------------------------------------------------+

WhisperDecoder::WhisperDecoder()
    : _waveEof(true) {
}

std::shared_ptr<WhisperDecoder> WhisperDecoder::create(
    std::shared_ptr<WhisperModel> model,
    std::shared_ptr<Wave> wave) {
  std::shared_ptr<WhisperDecoder> decoder{new WhisperDecoder()};

  // get timestamp token
  decoder->_chunkDecoder = WhisperChunkGreedySearchDecoder::fromModel(model);
  decoder->_wave = wave;

  return decoder;
}

std::optional<RecognitionResult> WhisperDecoder::nextResult() {
  while (_results.empty() && !_wave->eof()) {
    lut::Duration offset = _wave->tell();
    LOG(INFO) << "offset=" << offset.toString();
    Tensor waveChunk = _wave->read(lut::Duration::seconds(30));
    if (waveChunk.empty() || waveChunk.getShape(0) <= 1600) {
      // we ignore the chunks less than 0.1s (1600 samples)
      break;
    }

    std::vector<RecognitionResult> results = _chunkDecoder->decode(waveChunk);
    for (RecognitionResult &result : results) {
      result.begin += offset;
      result.end += offset;
    }

    if ((!_wave->eof()) && !results.empty()) {
      _wave->seek(results.back().end);
    }

    std::move(results.begin(), results.end(), std::back_inserter(_results));
  }

  if (_results.empty()) {
    return std::nullopt;
  } else {
    RecognitionResult result = _results.front();
    _results.pop_front();
    return result;
  }
}

}  // namespace whisper
}  // namespace libllm