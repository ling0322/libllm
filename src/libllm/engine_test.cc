// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "libllm/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/forward_batch.h"
#include "libllm/kv_cache.h"
#include "libllm/request.h"
#include "libllm/tokenizer.h"

namespace libllm {
namespace {

constexpr int VocabSize = 16;

class EngineTestVocab : public Vocab {
 public:
  EngineTestVocab() {
    for (int i = 0; i < VocabSize; ++i) _pieces.push_back(std::to_string(i));
  }

  int findToken(const std::string &) const override { return 0; }
  int findControlToken(const std::string &) const override { return 0; }
  const std::string &getTokenPiece(int tokenId) const override { return _pieces.at(tokenId); }
  const std::string &getTokenString(int tokenId) const override { return _pieces.at(tokenId); }
  int getVocabSize() const override { return VocabSize; }
  bool isControlToken(int) const override { return false; }
  int getUnkId() const override { return 0; }

 private:
  std::vector<std::string> _pieces;
};

class EngineTestTokenizer : public Tokenizer {
 public:
  std::vector<int> encode(const std::string &) const override { return {}; }
  const Vocab *getVocab() const override { return &_vocab; }

 private:
  EngineTestVocab _vocab;
};

class EngineTestModel : public ModelForGeneration {
 public:
  EngineTestModel() {
    _tokenizer = std::make_shared<EngineTestTokenizer>();
    _kvCacheManager = std::make_shared<KVCacheManager>(
        getKVCacheSpec(), 4, 16, fl::Device::getCpu());
  }

  fl::Tensor forward(const ForwardBatch &batch) const override {
    std::vector<float> logits(batch.numSequences() * VocabSize, -10.0f);
    lut::Span<const int> cuQ = batch.cuSeqlensQ();
    lut::Span<const fl::LongType> tokens = batch.tokenIds();
    for (int i = 0; i < batch.numSequences(); ++i) {
      int tokenId = static_cast<int>(tokens[cuQ[i + 1] - 1]);
      logits[i * VocabSize + (tokenId + 1) % VocabSize] = 10.0f;
    }
    return fl::Tensor::create<float>({batch.numSequences(), VocabSize}, logits);
  }

  bool isStopToken(int tokenId) const override { return tokenId == 15; }
  const char *getName() const override { return "engine-test"; }
  fl::Device getDevice() const override { return fl::Device::getCpu(); }
  int getOutputDim() const override { return VocabSize; }
  KVCacheSpec getKVCacheSpec() const override {
    return KVCacheSpec(1, 1, 2, 64, fl::DType::kFloat);
  }
  Prompt buildPrompt(lut::Span<const Message>) const override { return Prompt(); }
};

class SlowEngineTestModel : public EngineTestModel {
 public:
  fl::Tensor forward(const ForwardBatch &batch) const override {
    ++_forwardCalls;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return EngineTestModel::forward(batch);
  }

  int getForwardCalls() const { return _forwardCalls.load(); }

 private:
  mutable std::atomic<int> _forwardCalls{0};
};

GenerationConfig engineTestConfig(int maxTokens) {
  GenerationConfig config;
  config.temperature = 0.0f;
  config.topK = 0;
  config.topP = 1.0f;
  config.maxTokens = maxTokens;
  return config;
}

std::shared_ptr<Request> engineTestRequest(
    const std::string &requestId,
    fl::LongType tokenId,
    int maxTokens) {
  return std::make_shared<Request>(
      requestId,
      std::vector<fl::LongType>{tokenId},
      engineTestConfig(maxTokens));
}

}  // namespace

CATCH_TEST_CASE("Engine streams outputs and accepts callback requests", "[libllm][engine]") {
  auto model = std::make_shared<EngineTestModel>();
  std::mutex mutex;
  std::condition_variable outputReady;
  std::unordered_map<std::string, RequestOutput> finalOutputs;
  Engine *enginePointer = nullptr;
  bool addedFromCallback = false;
  bool callbackThreadConsistent = true;
  std::thread::id callbackThreadId;
  std::thread::id callerThreadId = std::this_thread::get_id();

  Engine::OutputCallback callback = [&](const std::vector<RequestOutput> &outputs) {
    bool addRequest = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (callbackThreadId == std::thread::id()) {
        callbackThreadId = std::this_thread::get_id();
      } else {
        callbackThreadConsistent &= callbackThreadId == std::this_thread::get_id();
      }
      for (const RequestOutput &output : outputs) {
        if (output.finished) finalOutputs[output.requestId] = output;
      }
      if (!addedFromCallback && !outputs.empty()) {
        addedFromCallback = true;
        addRequest = true;
      }
    }
    if (addRequest) enginePointer->addRequest(engineTestRequest("callback", 8, 1));
    outputReady.notify_all();
  };
  Engine engine(model, 4, std::move(callback));
  enginePointer = &engine;

  engine.addRequest(engineTestRequest("a", 1, 2));
  engine.addRequest(engineTestRequest("b", 5, 2));

  {
    std::unique_lock<std::mutex> lock(mutex);
    bool completed = outputReady.wait_for(
        lock,
        std::chrono::seconds(5),
        [&]() { return finalOutputs.size() == 3; });
    CATCH_REQUIRE(completed);
  }

  CATCH_REQUIRE(callbackThreadConsistent);
  CATCH_REQUIRE(callbackThreadId != callerThreadId);
  CATCH_REQUIRE(finalOutputs.at("a").finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(finalOutputs.at("b").finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(finalOutputs.at("callback").finishReason == RequestFinishReason::kLength);
  engine.shutdown();
}

CATCH_TEST_CASE("Engine shutdown cancels unfinished requests", "[libllm][engine]") {
  auto model = std::make_shared<EngineTestModel>();
  std::mutex mutex;
  std::vector<RequestOutput> outputs;

  Engine engine(model, 4, [&](const std::vector<RequestOutput> &batch) {
    std::lock_guard<std::mutex> lock(mutex);
    outputs.insert(outputs.end(), batch.begin(), batch.end());
  });
  engine.addRequest(engineTestRequest("long", 1, 100));
  engine.shutdown();

  std::lock_guard<std::mutex> lock(mutex);
  auto final = std::find_if(outputs.begin(), outputs.end(), [](const RequestOutput &output) {
    return output.requestId == "long" && output.finished;
  });
  CATCH_REQUIRE(final != outputs.end());
  // Whether it was cancelled or beat the shutdown to the finish line, it got a final output.
  CATCH_REQUIRE((
      final->finishReason == RequestFinishReason::kCancelled ||
      final->finishReason == RequestFinishReason::kLength));
  CATCH_REQUIRE_THROWS(engine.addRequest(engineTestRequest("late", 1, 1)));
}

CATCH_TEST_CASE("Engine delivers output while generation is active", "[libllm][engine]") {
  auto model = std::make_shared<SlowEngineTestModel>();
  std::mutex mutex;
  std::condition_variable outputReady;
  bool receivedIncrementalOutput = false;

  Engine engine(model, 4, [&](const std::vector<RequestOutput> &outputs) {
    std::lock_guard<std::mutex> lock(mutex);
    // A non-final output means the request is still running, so generation did not have to
    // complete before its first tokens were delivered.
    if (!outputs.empty() && !outputs.front().finished) receivedIncrementalOutput = true;
    outputReady.notify_all();
  });
  engine.addRequest(engineTestRequest("streaming", 1, 20));

  {
    std::unique_lock<std::mutex> lock(mutex);
    bool received = outputReady.wait_for(
        lock,
        std::chrono::seconds(2),
        [&]() { return receivedIncrementalOutput; });
    CATCH_REQUIRE(received);
  }
  CATCH_REQUIRE(model->getForwardCalls() < 20);
  engine.shutdown();
}

CATCH_TEST_CASE("Engine requires an output callback", "[libllm][engine]") {
  auto model = std::make_shared<EngineTestModel>();
  CATCH_REQUIRE_THROWS(Engine(model, 4, {}));
}

}  // namespace libllm
