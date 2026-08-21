// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include "libllm/scheduler.h"

#include <string.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "libllm/kv_cache.h"
#include "libllm/forward_batch.h"
#include "libllm/request.h"
#include "libllm/sampling_batch.h"
#include "flint/functional.h"

namespace libllm {

GenerationConfig::GenerationConfig()
  : topK(0),
      topP(1.0f),
  temperature(1.0f),
  maxTokens(std::numeric_limits<int>::max()) {
}

// -----------------------------------------------------------------------------------------------+
// class Sampler                                                                                  |
// -----------------------------------------------------------------------------------------------+

Sampler::Sampler(int topK, float topP)
    : _topK(topK),
      _topP(topP) {
}

std::vector<int> Sampler::getTopP(const fl::Tensor &distribution, lut::Span<const int> topK) {
  CHECK(distribution.getDim() == 1 && distribution.getDType() == fl::DType::kFloat);
  float sumP = 0.0f;

  std::vector<int> topP;
  const float *d = distribution.getInternalData()->getData<float>(distribution.getInternalOffset());
  for (int label : topK) {
    float p = d[label];
    topP.push_back(label);

    sumP += p;
    if (sumP >= _topP) {
      break;
    }
  }

  return topP;
}

std::vector<int> Sampler::getTopK(const fl::Tensor &distribution) {
  int topKCount = _topK <= 0 ? distribution.getShape(0) : _topK;
  CHECK(topKCount <= distribution.getShape(0) && distribution.getStride(0) == 1);
  if (_topBuffer.size() != distribution.getShape(0)) _topBuffer.resize(distribution.getShape(0));

  const float *d = distribution.getInternalData()->getData<float>(distribution.getInternalOffset());
  for (int32_t i = 0; i < distribution.getShape(0); ++i) {
    _topBuffer[i] = std::make_pair(i, d[i]);
  }

  std::partial_sort(
      _topBuffer.begin(),
      _topBuffer.begin() + topKCount,
      _topBuffer.end(),
      [](const std::pair<int32_t, float> &a, const std::pair<int32_t, float> &b) {
        return a.second > b.second;
      });

  std::vector<int> topK;
  LOG(DEBUG) << "Sampler TopK (K=" << topKCount << ")";
  for (int i = 0; i < topKCount; ++i) {
    topK.push_back(_topBuffer[i].first);
    LOG(DEBUG) << i << ": " << _topBuffer[i].first << ", " << _topBuffer[i].second;
  }

  return topK;
}

int Sampler::sampleTopP(const fl::Tensor &distribution, lut::Span<const int> topP) {
  CHECK(distribution.getDim() == 1 && distribution.getDType() == fl::DType::kFloat);
  std::vector<float> probAcc;

  float sumP = 0.0f;
  const float *probData = distribution.getInternalData()->getData<float>(
      distribution.getInternalOffset());
  for (int label : topP) {
    float p = probData[label];
    sumP += p;
    probAcc.push_back(sumP);
  }

  float r = _random.nextFloat() * sumP;
  for (int i = 0; i < topP.size(); ++i) {
    if (r < probAcc[i]) {
      return topP[i];
    }
  }
  return topP.back();
}

int Sampler::sample(const fl::Tensor &distribution) {
  int topKCount = _topK <= 0 ? distribution.getShape(-1) : _topK;
  if (distribution.getDevice().getType() == fl::Device::kCuda) {
    fl::Tensor sampled = fl::F::sample(distribution, topKCount, _topP);
    sampled = fl::F::to(fl::Device::getCpu(), sampled);
    CHECK(sampled.getNumEl() == 1 && sampled.getDType() == fl::DType::kLong);
    return static_cast<int>(*sampled.getInternalData()->getData<fl::LongType>(
        sampled.getInternalOffset()));
  }
  CHECK(distribution.getDim() == 1 && distribution.getDType() == fl::DType::kFloat);

  std::vector<int> topK = getTopK(distribution);  // topK is sorted by its prob in x
  std::vector<int> topP = getTopP(distribution, topK);

  return sampleTopP(distribution, topP);
}

int Sampler::greedy(fl::Tensor logits) {
  CHECK(logits.getDim() == 1);
  if (logits.getDevice().getType() == fl::Device::kCuda) {
    fl::Tensor sampled = fl::F::sample(logits, 1, 1.0f);
    sampled = fl::F::to(fl::Device::getCpu(), sampled);
    return static_cast<int>(*sampled.getInternalData()->getData<fl::LongType>(
        sampled.getInternalOffset()));
  }
  if (logits.getDType() == fl::DType::kFloat16) {
    logits = fl::F::cast(logits, fl::DType::kFloat);
  }
  CHECK(logits.getDType() == fl::DType::kFloat && logits.getStride(0) == 1);
  const float *data = logits.getInternalData()->getData<float>(logits.getInternalOffset());
  return static_cast<int>(std::max_element(data, data + logits.getShape(0)) - data);
}

// -----------------------------------------------------------------------------------------------+
// class Scheduler                                                                                |
// -----------------------------------------------------------------------------------------------+

Scheduler::Scheduler(const GenerationConfig &config, std::shared_ptr<ModelForGeneration> model)
    : _model(model),
      _currentToken(-1),
      _sampler(config.topK, config.topP),
      _temperature(config.temperature) {
}

std::shared_ptr<Scheduler> Scheduler::create(
    const GenerationConfig &config,
    std::shared_ptr<ModelForGeneration> model) {
  return std::shared_ptr<Scheduler>{new Scheduler(config, model)};
}

bool Scheduler::generate() {
  if (_model->isStopToken(_currentToken)) return false;

  lut::Span<const fl::LongType> tokenIds = _request->getTokenIds();
  int contextLength = _request->getContextLength();
  int numTokensToCompute = static_cast<int>(tokenIds.size()) - contextLength;
  CHECK(numTokensToCompute > 0);

  reserveBlocks(contextLength + numTokensToCompute);

  ForwardBatch batch = ForwardBatch::single(
      tokenIds.subspan(contextLength, numTokensToCompute),
      contextLength);
  batch.setKVCacheManager(_model->getKVCacheManager());
  lut::Span<const int> blockIds = _request->getBlockIds();
  batch.setBlockIds({std::vector<int>(blockIds.begin(), blockIds.end())});
  batch.prepare(_model->getDevice());

  _currentToken = searchToken(_model->forward(batch));

  _request->advanceComputedTokens(numTokensToCompute);
  _request->setContextLength(contextLength + numTokensToCompute);

  LOG(DEBUG) << lut::sprintf(
      "%d -> \"%s\"",
      _currentToken,
      _model->getVocab()->getTokenString(_currentToken));
  if (_model->isStopToken(_currentToken)) {
    _request->finish();
    releaseBlocks();
    return false;
  }

  _request->appendToken(_currentToken);
  return true;
}

void Scheduler::reserveBlocks(int numTokens) {
  std::shared_ptr<KVCacheManager> manager = _model->getKVCacheManager().lock();
  CHECK(manager) << "model has no KV cache manager";

  int numBlocksHeld = static_cast<int>(_request->getBlockIds().size());
  int numBlocksNeeded = manager->getNumBlocksForTokens(numTokens);
  if (numBlocksNeeded <= numBlocksHeld) return;

  std::vector<int> blockIds = manager->allocateBlocks(numBlocksNeeded - numBlocksHeld);
  if (blockIds.empty()) {
    throw lut::AbortedError("KV cache is full");
  }

  _request->addBlockIds(blockIds);
}

void Scheduler::releaseBlocks() {
  if (_request->getBlockIds().empty()) return;

  std::shared_ptr<KVCacheManager> manager = _model->getKVCacheManager().lock();
  if (manager) manager->freeBlocks(_request->getBlockIds());
  _request->clearBlockIds();
}

Scheduler::~Scheduler() {
  // A caller that stops early still owes the manager its blocks.
  if (_request) releaseBlocks();
}

void Scheduler::setRequest(std::shared_ptr<Request> request) {
  CHECK(request);
  _request = request;
  _request->setStatus(RequestStatus::kRunning);
}

std::string Scheduler::getToken() {
  if (_currentToken < 0) return "";

  const Vocab *vocab = _model->getVocab();
  const char *token = vocab->getTokenPiece(_currentToken).c_str();
  return token;
}

std::string Scheduler::getTokenName() {
  if (_currentToken < 0) return "";

  const Vocab *vocab = _model->getVocab();
  const char *token = vocab->getTokenString(_currentToken).c_str();
  return token;
}

int Scheduler::searchToken(const fl::Tensor &logits) {
  CHECK(logits.getDim() == 2 && logits.getShape(0) == 1);

  fl::Tensor x = logits.subtensor(0);
  if (_temperature == 0.0f) return _sampler.greedy(x);
  if (_temperature != 1.0f) {
    x = fl::F::mul(x, 1.0f / _temperature);
  }

  x = fl::F::softmax(x);
  if (x.getDevice().getType() == fl::Device::kCuda) {
    return _sampler.sample(x);
  }
  if (x.getDType() == fl::DType::kFloat16) {
    x = fl::F::cast(x, fl::DType::kFloat);
  }

  return _sampler.sample(x);
}

// -----------------------------------------------------------------------------------------------+
// class SchedulerV2                                                                              |
// -----------------------------------------------------------------------------------------------+

struct SchedulerV2::ScheduledBatch {
  ScheduledBatch(
      ForwardBatch batch,
      std::vector<Request *> requests,
      std::vector<int> queryLengths,
      SamplingBatch samplingBatch)
      : batch(std::move(batch)),
        requests(std::move(requests)),
        queryLengths(std::move(queryLengths)),
        samplingBatch(std::move(samplingBatch)) {
  }

  ForwardBatch batch;
  std::vector<Request *> requests;
  std::vector<int> queryLengths;
  SamplingBatch samplingBatch;
};

SchedulerV2::SchedulerV2(
    std::shared_ptr<ModelForGeneration> model,
    int maxNumBatchedTokens)
    : _model(std::move(model)),
      _maxNumBatchedTokens(maxNumBatchedTokens) {
  if (!_model) throw lut::AbortedError("scheduler requires a model");
  if (_maxNumBatchedTokens <= 0) {
    throw lut::AbortedError("max_num_batched_tokens must be positive");
  }
  if (_model->getKVCacheManager().expired()) {
    throw lut::AbortedError("model has no KV cache manager");
  }
}

SchedulerV2::~SchedulerV2() {
  for (auto &entry : _requests) releaseBlocks(*entry.second);
}

void SchedulerV2::addRequest(std::shared_ptr<Request> request) {
  if (!request) throw lut::AbortedError("cannot add a null request");
  if (request->getId().empty()) throw lut::AbortedError("request id is empty");
  if (_requests.find(request->getId()) != _requests.end()) {
    throw lut::AbortedError("duplicate request id: " + request->getId());
  }
  if (request->isFinished()) throw lut::AbortedError("request is already finished");

  const GenerationConfig &config = request->getConfig();
  if (!std::isfinite(config.temperature) || config.temperature < 0.0f) {
    throw lut::AbortedError("temperature must be finite and not negative");
  }
  if (config.topK < -1 || config.topK > _model->getOutputDim()) {
    throw lut::AbortedError("top_k is out of range");
  }
  if (!std::isfinite(config.topP) || config.topP <= 0.0f || config.topP > 1.0f) {
    throw lut::AbortedError("top_p is out of range");
  }
  if (config.maxTokens < 0) throw lut::AbortedError("max_tokens must not be negative");

  std::string requestId = request->getId();
  request->setStatus(RequestStatus::kWaiting);
  if (config.maxTokens == 0) {
    request->setPendingFinishReason(RequestFinishReason::kLength);
  }
  _requestOrder.push_back(requestId);
  _waitingOrder.push_back(requestId);
  _requests.emplace(std::move(requestId), std::move(request));
}

void SchedulerV2::abortRequest(const std::string &requestId) {
  auto it = _requests.find(requestId);
  if (it == _requests.end() || it->second->isFinalEmitted()) return;
  it->second->setPendingFinishReason(RequestFinishReason::kCancelled);
  it->second->setErrorMessage("");
}

bool SchedulerV2::hasUnfinishedRequests() const {
  return !_requests.empty();
}

int SchedulerV2::getNumUnfinishedRequests() const {
  return static_cast<int>(_requests.size());
}

bool SchedulerV2::reserveBlocks(Request &request, int numTokens) {
  std::shared_ptr<KVCacheManager> manager = _model->getKVCacheManager().lock();
  CHECK(manager) << "model has no KV cache manager";

  int numBlocksNeeded = manager->getNumBlocksForTokens(numTokens);
  if (numTokens > _model->getKVCacheSpec().getMaxContextLength() ||
      numBlocksNeeded > manager->getMaxNumBlocksPerRequest() ||
      numBlocksNeeded > manager->getNumBlocks()) {
    request.setPendingFinishReason(RequestFinishReason::kError);
    request.setErrorMessage("request exceeds the KV cache context capacity");
    return false;
  }

  int numBlocksHeld = static_cast<int>(request.getBlockIds().size());
  if (numBlocksNeeded <= numBlocksHeld) return true;

  std::vector<int> blocks = manager->allocateBlocks(numBlocksNeeded - numBlocksHeld);
  if (blocks.empty()) return false;
  request.addBlockIds(blocks);
  return true;
}

void SchedulerV2::preemptRequest(Request &request) {
  releaseBlocks(request);
  request.resetComputedTokens();
  request.setStatus(RequestStatus::kPreempted);

  const std::string &requestId = request.getId();
  auto runningIt = std::find(_runningOrder.begin(), _runningOrder.end(), requestId);
  CHECK(runningIt != _runningOrder.end());
  _runningOrder.erase(runningIt);
  _waitingOrder.push_front(requestId);
}

Request *SchedulerV2::selectPreemptionVictim(
    const std::vector<Request *> &scheduledRequests) const {
  for (auto it = _runningOrder.rbegin(); it != _runningOrder.rend(); ++it) {
    Request &candidate = *_requests.at(*it);
    if (candidate.getPendingFinishReason() != RequestFinishReason::kNone ||
      candidate.isFinalEmitted() ||
        std::find(scheduledRequests.begin(), scheduledRequests.end(), &candidate) !=
            scheduledRequests.end()) {
      continue;
    }
    return &candidate;
  }
  return nullptr;
}

void SchedulerV2::releaseBlocks(Request &request) {
  lut::Span<const int> blockIds = request.getBlockIds();
  if (blockIds.empty()) return;

  std::shared_ptr<KVCacheManager> manager = _model->getKVCacheManager().lock();
  if (manager) manager->freeBlocks(blockIds);
  request.clearBlockIds();
}

std::unique_ptr<SchedulerV2::ScheduledBatch> SchedulerV2::schedule() {
  // Share one token budget across all requests selected for this forward pass.
  int tokenBudget = _maxNumBatchedTokens;

  // These arrays remain in the same sequence order. ScheduledBatch uses the state and query
  // metadata after the model returns, while ForwardBatch owns the packed model inputs.
  std::vector<Request *> requests;
  std::vector<int> queryLengths;
  std::vector<fl::LongType> sampleSequenceIndices;
  std::vector<float> temperatures;
  std::vector<fl::IntType> topKs;
  std::vector<float> topPs;
  std::vector<fl::LongType> tokenIds;
  std::vector<fl::LongType> positionIds;

  // FlashAttention represents ragged query and key lengths as exclusive prefix sums.
  std::vector<int> cuSeqlensQ{0};
  std::vector<int> cuSeqlensK{0};
  std::vector<std::vector<int>> blockIds;

  // Running requests retain scheduling priority over waiting requests. Snapshot the candidates
  // and boundary so queue transitions below do not change this step's admission class.
  std::vector<std::string> candidates(_runningOrder.begin(), _runningOrder.end());
  int numRunningCandidates = static_cast<int>(candidates.size());
  candidates.insert(candidates.end(), _waitingOrder.begin(), _waitingOrder.end());
  std::vector<std::string> preemptedIds;

  // Walk running requests first, followed by the FCFS waiting queue.
  for (int candidateIndex = 0;
       candidateIndex < static_cast<int>(candidates.size());
       ++candidateIndex) {
    // No more requests can be admitted once this forward pass has consumed its token budget.
    if (tokenBudget == 0) break;
    const std::string &requestId = candidates[candidateIndex];
    bool wasRunning = candidateIndex < numRunningCandidates;
    Request &request = *_requests.at(requestId);

    // A lower-priority running request may have been preempted while making room for an earlier
    // candidate. It remains ineligible for this step, then resumes from the waiting queue on the
    // next one.
    if (std::find(preemptedIds.begin(), preemptedIds.end(), requestId) != preemptedIds.end()) {
      continue;
    }

    // Match vLLM admission behavior: once this step preempts a running request, preserve the
    // released capacity for the surviving running batch and defer every waiting request.
    if (!wasRunning && !preemptedIds.empty()) break;

    // Final/error/cancelled requests are emitted outside schedule() and must not enter a model
    // batch.
    if (request.getPendingFinishReason() != RequestFinishReason::kNone ||
      request.isFinalEmitted()) {
      continue;
    }

    // contextLength is the KV prefix already computed. The remaining request tokens are either
    // an unfinished prompt chunk or the single generated token appended by the previous step.
    lut::Span<const fl::LongType> requestTokens = request.getTokenIds();
    int contextLength = request.getContextLength();
    int numUncomputedTokens = static_cast<int>(requestTokens.size()) - contextLength;
    if (numUncomputedTokens <= 0) {
      request.setPendingFinishReason(RequestFinishReason::kError);
      request.setErrorMessage("request has no token to compute");
      continue;
    }

    // Chunk long prefills to fit the remaining budget. Decode requests naturally have one
    // uncomputed token and therefore consume one budget slot.
    int queryLength = std::min(numUncomputedTokens, tokenBudget);

    // A running request has priority over lower-priority running requests. Evict victims from the
    // tail and retry until this request fits. If it is its own last possible victim, defer it and
    // recompute from the retained token history in a later step.
    bool blocksReserved = reserveBlocks(request, contextLength + queryLength);
    while (!blocksReserved &&
        request.getPendingFinishReason() == RequestFinishReason::kNone &&
          wasRunning) {
      Request *victim = selectPreemptionVictim(requests);
      if (!victim) break;

      bool preemptedCurrentRequest = victim == &request;
      std::string victimId = victim->getId();
      preemptRequest(*victim);
      preemptedIds.push_back(std::move(victimId));
      if (preemptedCurrentRequest) break;

      blocksReserved = reserveBlocks(request, contextLength + queryLength);
    }

    if (!blocksReserved) {
      // A permanently oversized request already carries kError. A waiting request remains at the
      // queue head until running requests release enough blocks. A self-preempted request also
      // waits for the next step so it cannot immediately reclaim the blocks it just released.
      if (request.getPendingFinishReason() != RequestFinishReason::kNone) continue;
      if (!wasRunning ||
          std::find(preemptedIds.begin(), preemptedIds.end(), requestId) !=
              preemptedIds.end()) {
        break;
      }
      continue;
    }

    if (!wasRunning) {
      auto waitingIt = std::find(_waitingOrder.begin(), _waitingOrder.end(), requestId);
      CHECK(waitingIt != _waitingOrder.end());
      _waitingOrder.erase(waitingIt);
      _runningOrder.push_back(requestId);
      request.setStatus(RequestStatus::kRunning);
    }

    // Append this sequence's query tokens and their absolute rotary/cache positions to the packed
    // token dimension.
    lut::Span<const fl::LongType> queryTokens =
        requestTokens.subspan(contextLength, queryLength);
    tokenIds.insert(tokenIds.end(), queryTokens.begin(), queryTokens.end());
    for (int i = 0; i < queryLength; ++i) positionIds.push_back(contextLength + i);

    // Keep per-sequence execution metadata aligned with the packed ForwardBatch sequence order.
    requests.push_back(&request);
    queryLengths.push_back(queryLength);

    // Intermediate prompt chunks update KV only. A completed prompt/decode row contributes one
    // set of per-request parameters to the batched sampler.
    if (queryLength == numUncomputedTokens) {
      const GenerationConfig &config = request.getConfig();
      sampleSequenceIndices.push_back(static_cast<fl::LongType>(requests.size() - 1));
      temperatures.push_back(config.temperature);
      topKs.push_back(static_cast<fl::IntType>(config.topK));
      topPs.push_back(config.topP);
    }

    // Q contains only this step's tokens; K includes both the cached prefix and the new query.
    cuSeqlensQ.push_back(cuSeqlensQ.back() + queryLength);
    cuSeqlensK.push_back(cuSeqlensK.back() + contextLength + queryLength);

    // ForwardBatch turns each logical block list into the paged-attention block-table row.
    lut::Span<const int> requestBlocks = request.getBlockIds();
    blockIds.emplace_back(requestBlocks.begin(), requestBlocks.end());
    tokenBudget -= queryLength;
  }

  if (requests.empty()) return nullptr;

  // Materialize the ragged packed batch and its paged-cache addressing tensors once, before the
  // model reads it. The CPU vectors above are moved because ScheduledBatch only needs the prepared
  // ForwardBatch and its per-sequence execution metadata.
  ForwardBatch batch = ForwardBatch::packed(
      std::move(tokenIds),
      std::move(cuSeqlensQ),
      std::move(cuSeqlensK),
      std::move(positionIds));
  batch.setKVCacheManager(_model->getKVCacheManager());
  batch.setBlockIds(std::move(blockIds));
  batch.prepare(_model->getDevice());
  SamplingBatch samplingBatch(
      std::move(sampleSequenceIndices),
      std::move(temperatures),
      std::move(topKs),
      std::move(topPs));
  samplingBatch.prepare(_model->getDevice());
  return std::make_unique<ScheduledBatch>(
      std::move(batch),
      std::move(requests),
      std::move(queryLengths),
      std::move(samplingBatch));
}

std::vector<RequestOutput> SchedulerV2::finishCancelledRequests() {
  std::vector<RequestOutput> outputs;
  for (const std::string &requestId : _requestOrder) {
    Request &request = *_requests.at(requestId);
    if (request.isFinalEmitted() ||
      request.getPendingFinishReason() == RequestFinishReason::kNone) {
      continue;
    }

    RequestOutput output;
    output.requestId = requestId;
    output.finished = true;
    output.finishReason = request.getPendingFinishReason();
    output.errorMessage = request.getErrorMessage();
    request.finish();
    releaseBlocks(request);
    request.setFinalEmitted(true);
    outputs.push_back(std::move(output));
  }
  return outputs;
}

std::vector<RequestOutput> SchedulerV2::execute(ScheduledBatch &batch) {
  fl::Tensor logits;
  try {
    logits = _model->forward(batch.batch);
    CHECK(logits.getDim() == 2);
    CHECK(logits.getShape(0) == static_cast<int>(batch.requests.size()));
  } catch (const std::exception &error) {
    for (Request *request : batch.requests) {
      request->setPendingFinishReason(RequestFinishReason::kError);
      request->setErrorMessage(error.what());
    }
    return finishCancelledRequests();
  }

  fl::Tensor sampledTokenIds;
  if (!batch.samplingBatch.empty()) {
    try {
      sampledTokenIds = batch.samplingBatch.sample(logits);
      if (sampledTokenIds.getDevice().getType() != fl::Device::kCpu) {
        sampledTokenIds = fl::F::to(fl::Device::getCpu(), sampledTokenIds);
      }
    } catch (const std::exception &error) {
      for (Request *request : batch.requests) {
        request->setPendingFinishReason(RequestFinishReason::kError);
        request->setErrorMessage(error.what());
      }
      return finishCancelledRequests();
    }
  }

  std::vector<RequestOutput> outputs;
  for (int i = 0; i < static_cast<int>(batch.requests.size()); ++i) {
    Request &request = *batch.requests[i];
    int queryLength = batch.queryLengths[i];
    int contextLength = request.getContextLength() + queryLength;
    request.advanceComputedTokens(queryLength);
    request.setContextLength(contextLength);
  }

  const fl::LongType *sampledData = batch.samplingBatch.empty()
      ? nullptr
      : sampledTokenIds.getInternalData()->getData<fl::LongType>(
            sampledTokenIds.getInternalOffset());
  const std::vector<fl::LongType> &sampleIndices = batch.samplingBatch.sequenceIndices();
  for (int sampleIndex = 0; sampleIndex < batch.samplingBatch.size(); ++sampleIndex) {
    int sequenceIndex = static_cast<int>(sampleIndices[sampleIndex]);
    Request &request = *batch.requests[sequenceIndex];
    const GenerationConfig &config = request.getConfig();
    int tokenId = static_cast<int>(sampledData[sampleIndex]);

    RequestOutput output;
    output.requestId = request.getId();
    if (_model->isStopToken(tokenId)) {
      output.finished = true;
      output.finishReason = RequestFinishReason::kStop;
    } else {
      request.appendToken(tokenId);
      output.tokenIds.push_back(tokenId);
      output.text = _model->getVocab()->getTokenPiece(tokenId);
      if (request.getNumGeneratedTokens() >= config.maxTokens) {
        output.finished = true;
        output.finishReason = RequestFinishReason::kLength;
      }
    }

    if (output.finished) {
      request.finish();
      releaseBlocks(request);
      request.setFinalEmitted(true);
    }
    outputs.push_back(std::move(output));
  }
  return outputs;
}

void SchedulerV2::removeFinishedRequests() {
  auto it = _requestOrder.begin();
  while (it != _requestOrder.end()) {
    auto requestIt = _requests.find(*it);
    if (requestIt != _requests.end() && requestIt->second->isFinalEmitted()) {
      _runningOrder.erase(
          std::remove(_runningOrder.begin(), _runningOrder.end(), *it),
          _runningOrder.end());
      _waitingOrder.erase(
          std::remove(_waitingOrder.begin(), _waitingOrder.end(), *it),
          _waitingOrder.end());
      _requests.erase(requestIt);
      it = _requestOrder.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<RequestOutput> SchedulerV2::step() {
  std::vector<RequestOutput> outputs = finishCancelledRequests();
  removeFinishedRequests();

  std::unique_ptr<ScheduledBatch> batch = schedule();
  std::vector<RequestOutput> schedulingErrors = finishCancelledRequests();
  outputs.insert(
      outputs.end(),
      std::make_move_iterator(schedulingErrors.begin()),
      std::make_move_iterator(schedulingErrors.end()));
  removeFinishedRequests();

  if (batch) {
    std::vector<RequestOutput> generated = execute(*batch);
    outputs.insert(
        outputs.end(),
        std::make_move_iterator(generated.begin()),
        std::make_move_iterator(generated.end()));
    removeFinishedRequests();
  }
  return outputs;
}

}  // namespace libllm
