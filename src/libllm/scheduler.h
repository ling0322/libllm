// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "libllm/model_for_generation.h"
#include "libllm/prompt.h"
#include "lutil/random.h"

namespace libllm {

// request.h needs GenerationConfig from this header, so it stays a forward declaration here.
class Request;

struct GenerationConfig {
  int topK;
  float topP;
  float temperature;
  int maxTokens;

  GenerationConfig();
};

class Sampler {
 public:
  Sampler(int topK, float topP);

  int sample(const fl::Tensor &distribution);
  int greedy(fl::Tensor logits);

 private:
  lut::Random _random;
  int _topK;
  float _topP;
  std::vector<std::pair<int, float>> _topBuffer;

  std::vector<int> getTopK(const fl::Tensor &distribution);
  std::vector<int> getTopP(const fl::Tensor &distribution, lut::Span<const int> topK);
  int sampleTopP(const fl::Tensor &distribution, lut::Span<const int> topP);
};

/// @brief Drives a model through one prompt, handing out one sampled token at a time.
class Scheduler {
 public:
  static std::shared_ptr<Scheduler> create(
      const GenerationConfig &config,
      std::shared_ptr<ModelForGeneration> model);

  ~Scheduler();

  /// @brief set the request to complete.
  /// @param request the request;
  void setRequest(std::shared_ptr<Request> request);

  /// @brief generate next token. Return false if generation is finished.
  /// @return if generation is finished.
  bool generate();

  /// @brief get the piece of current token.
  /// @return piece of current token.
  std::string getToken();

  /// @brief get the display name of current token.
  /// @return name of current token.
  std::string getTokenName();

 private:
  std::shared_ptr<Request> _request;
  std::shared_ptr<ModelForGeneration> _model;
  int _currentToken;

  Sampler _sampler;
  float _temperature;

  Scheduler(const GenerationConfig &config, std::shared_ptr<ModelForGeneration> model);

  /// @brief Make sure the request holds enough blocks for `numTokens` tokens, allocating more if
  /// it does not. Throws when the manager has none left.
  void reserveBlocks(int numTokens);

  /// @brief Return the blocks of the request to the manager.
  void releaseBlocks();

  int searchToken(const fl::Tensor &logits);
};

/// Why a request stopped producing tokens.
enum class RequestFinishReason {
  kNone,
  kStop,
  kLength,
  kCancelled,
  kError,
};

/// One request's delta output from a SchedulerV2 step.
struct RequestOutput {
  std::string requestId;
  std::vector<fl::LongType> tokenIds;
  std::string text;
  bool finished = false;
  RequestFinishReason finishReason = RequestFinishReason::kNone;
  std::string errorMessage;
};

/// Synchronous continuous-batching scheduler core.
///
/// SchedulerV2 owns all accepted requests and advances a selected set through one packed model
/// forward per step. It does not own threads, parse JSON, or invoke stream callbacks; the Engine
/// serializes calls to this class and publishes the returned RequestOutput values.
class SchedulerV2 {
 public:
  /// `maxNumBatchedTokens` is the total query-token budget of one model forward.
  SchedulerV2(std::shared_ptr<ModelForGeneration> model, int maxNumBatchedTokens);
  ~SchedulerV2();

  SchedulerV2(const SchedulerV2 &) = delete;
  SchedulerV2 &operator=(const SchedulerV2 &) = delete;

  /// Accept a request into the waiting queue. Throws if its id is empty or already active.
  void addRequest(std::shared_ptr<Request> request);

  /// Mark a request for cancellation. Unknown and already finished ids are successful no-ops.
  /// The final kCancelled output is returned by a subsequent step().
  void abortRequest(const std::string &requestId);

  /// Mark every unfinished request for cancellation.
  void abortAllRequests();

  /// Schedule and execute at most one packed model forward.
  ///
  /// The result contains DELTA outputs: each entry holds only tokens and text produced since that
  /// request's previous output. A final entry is emitted exactly once for every accepted request.
  /// A step may return only cancellation/error outputs without executing the model.
  std::vector<RequestOutput> step();

  /// Return true while an accepted request still needs a final output.
  bool hasUnfinishedRequests() const;

  /// Number of waiting or running requests, including requests pending cancellation delivery.
  int getNumUnfinishedRequests() const;

 private:
  struct ScheduledBatch;

  std::shared_ptr<ModelForGeneration> _model;
  int _maxNumBatchedTokens;

  // The map owns requests. A request belongs to exactly one scheduling queue; request order
  // preserves deterministic output and cleanup order.
  std::unordered_map<std::string, std::shared_ptr<Request>> _requests;
  std::deque<std::string> _requestOrder;
  std::deque<std::string> _runningOrder;
  std::deque<std::string> _waitingOrder;

  std::unique_ptr<ScheduledBatch> schedule();
  std::vector<RequestOutput> execute(ScheduledBatch &batch);
  std::vector<RequestOutput> finishCancelledRequests();
  bool reserveBlocks(Request &request, int numTokens);
  Request *selectPreemptionVictim(const std::vector<Request *> &scheduledRequests) const;
  void preemptRequest(Request &request);
  void releaseBlocks(Request &request);
  void removeFinishedRequests();
};

}  // namespace libllm
