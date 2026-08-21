// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "libllm/request.h"

#include <utility>

#include "lutil/error.h"
#include "lutil/log.h"

namespace libllm {

Request::Request(
    std::string requestId,
    std::vector<fl::LongType> promptTokenIds,
    const GenerationConfig &config)
    : _id(std::move(requestId)),
      _config(config),
      _tokenIds(std::move(promptTokenIds)),
      _promptLength(static_cast<int>(_tokenIds.size())),
      _numComputedTokens(0),
      _contextLength(0),
      _status(RequestStatus::kWaiting),
      _pendingFinishReason(RequestFinishReason::kNone),
      _finalEmitted(false) {
  if (_tokenIds.empty()) {
    throw lut::AbortedError("request with an empty prompt");
  }
}

const std::string &Request::getId() const {
  return _id;
}

const GenerationConfig &Request::getConfig() const {
  return _config;
}

RequestStatus Request::getStatus() const {
  return _status;
}

void Request::setStatus(RequestStatus status) {
  _status = status;
}

int Request::getNumGeneratedTokens() const {
  return static_cast<int>(_tokenIds.size()) - _promptLength;
}

RequestFinishReason Request::getPendingFinishReason() const {
  return _pendingFinishReason;
}

void Request::setPendingFinishReason(RequestFinishReason finishReason) {
  _pendingFinishReason = finishReason;
}

const std::string &Request::getErrorMessage() const {
  return _errorMessage;
}

void Request::setErrorMessage(std::string errorMessage) {
  _errorMessage = std::move(errorMessage);
}

bool Request::isFinalEmitted() const {
  return _finalEmitted;
}

void Request::setFinalEmitted(bool finalEmitted) {
  _finalEmitted = finalEmitted;
}

lut::Span<const fl::LongType> Request::getTokenIds() const {
  return _tokenIds;
}

int Request::getNumComputedTokens() const {
  return _numComputedTokens;
}

void Request::advanceComputedTokens(int numTokens) {
  CHECK(numTokens >= 0 && _numComputedTokens + numTokens <= static_cast<int>(_tokenIds.size()));
  _numComputedTokens += numTokens;
}

void Request::resetComputedTokens() {
  _numComputedTokens = 0;
  _contextLength = 0;
}

int Request::getContextLength() const {
  return _contextLength;
}

void Request::setContextLength(int contextLength) {
  CHECK(contextLength >= 0 && contextLength <= static_cast<int>(_tokenIds.size()));
  _contextLength = contextLength;
}

lut::Span<const int> Request::getBlockIds() const {
  return _blockIds;
}

void Request::addBlockIds(lut::Span<const int> blockIds) {
  _blockIds.insert(_blockIds.end(), blockIds.begin(), blockIds.end());
}

void Request::clearBlockIds() {
  _blockIds.clear();
}

void Request::appendToken(fl::LongType tokenId) {
  _tokenIds.push_back(tokenId);
}

bool Request::isFinished() const {
  return _status == RequestStatus::kFinished;
}

void Request::finish() {
  _status = RequestStatus::kFinished;
}

}  // namespace libllm
