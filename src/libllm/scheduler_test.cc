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

#include "libllm/scheduler.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/forward_batch.h"
#include "libllm/request.h"
#include "libllm/tokenizer.h"

namespace libllm {
namespace {

constexpr int VocabSize = 16;
constexpr int BlockSize = 4;
constexpr int NumBlocks = 16;

class FakeVocab : public Vocab {
 public:
  FakeVocab() {
    for (int i = 0; i < VocabSize; ++i) pieces.push_back(std::to_string(i));
  }

  int findToken(const std::string &) const override { return 0; }
  int findControlToken(const std::string &) const override { return 0; }
  const std::string &getTokenPiece(int tokenId) const override { return pieces.at(tokenId); }
  const std::string &getTokenString(int tokenId) const override { return pieces.at(tokenId); }
  int getVocabSize() const override { return VocabSize; }
  bool isControlToken(int) const override { return false; }
  int getUnkId() const override { return 0; }

 private:
  std::vector<std::string> pieces;
};

class FakeTokenizer : public Tokenizer {
 public:
  std::vector<int> encode(const std::string &) const override { return {}; }
  const Vocab *getVocab() const override { return &vocab; }

 private:
  FakeVocab vocab;
};

struct BatchRecord {
  std::vector<int> queryLengths;
  std::vector<int> keyLengths;
  std::vector<fl::LongType> tokenIds;
  std::vector<fl::LongType> positionIds;
};

class FakeModel : public ModelForGeneration {
 public:
  explicit FakeModel(int numBlocks = NumBlocks) {
    _tokenizer = std::make_shared<FakeTokenizer>();
    _kvCacheManager = std::make_shared<KVCacheManager>(
        getKVCacheSpec(), BlockSize, numBlocks, fl::Device::getCpu());
  }

  fl::Tensor forward(const ForwardBatch &batch) const override {
    BatchRecord record;
    lut::Span<const int> cuQ = batch.cuSeqlensQ();
    lut::Span<const int> cuK = batch.cuSeqlensK();
    for (int i = 0; i < batch.numSequences(); ++i) {
      record.queryLengths.push_back(cuQ[i + 1] - cuQ[i]);
      record.keyLengths.push_back(cuK[i + 1] - cuK[i]);
    }
    record.tokenIds.assign(batch.tokenIds().begin(), batch.tokenIds().end());
    record.positionIds.assign(batch.positionIds().begin(), batch.positionIds().end());
    records.push_back(std::move(record));

    std::vector<float> logits(batch.numSequences() * VocabSize, -10.0f);
    lut::Span<const fl::LongType> tokens = batch.tokenIds();
    for (int i = 0; i < batch.numSequences(); ++i) {
      int lastQuery = cuQ[i + 1] - 1;
      int nextToken = (static_cast<int>(tokens[lastQuery]) + 1) % VocabSize;
      logits[i * VocabSize + nextToken] = 10.0f;
    }
    return fl::Tensor::create<float>({batch.numSequences(), VocabSize}, logits);
  }

  bool isStopToken(int tokenId) const override { return tokenId == 9; }
  const char *getName() const override { return "fake"; }
  fl::Device getDevice() const override { return fl::Device::getCpu(); }
  int getOutputDim() const override { return VocabSize; }
  KVCacheSpec getKVCacheSpec() const override {
    return KVCacheSpec(1, 1, 2, 32, fl::DType::kFloat);
  }
  Prompt buildPrompt(lut::Span<const Message>) const override { return Prompt(); }

  std::shared_ptr<KVCacheManager> cache() const { return _kvCacheManager; }

  mutable std::vector<BatchRecord> records;
};

GenerationConfig config(int maxTokens) {
  GenerationConfig result;
  result.topK = 1;
  result.topP = 1.0f;
  result.temperature = 1.0f;
  result.maxTokens = maxTokens;
  return result;
}

std::shared_ptr<Request> request(
    const std::string &id,
    std::vector<fl::LongType> prompt,
    int maxTokens) {
  return std::make_shared<Request>(id, std::move(prompt), config(maxTokens));
}

}  // namespace

CATCH_TEST_CASE("SchedulerV2 batches prefill and decode", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>();
  SchedulerV2 scheduler(model, 4);
  scheduler.addRequest(request("a", {1, 2, 3}, 2));
  scheduler.addRequest(request("b", {5}, 2));

  std::vector<RequestOutput> first = scheduler.step();
  CATCH_REQUIRE(first.size() == 2);
  CATCH_REQUIRE(first[0].requestId == "a");
  CATCH_REQUIRE(first[0].tokenIds == std::vector<fl::LongType>{4});
  CATCH_REQUIRE(first[0].text == "4");
  CATCH_REQUIRE_FALSE(first[0].finished);
  CATCH_REQUIRE(first[1].requestId == "b");
  CATCH_REQUIRE(first[1].tokenIds == std::vector<fl::LongType>{6});
  CATCH_REQUIRE(model->records.size() == 1);
  CATCH_REQUIRE(model->records[0].queryLengths == std::vector<int>{3, 1});
  CATCH_REQUIRE(model->records[0].keyLengths == std::vector<int>{3, 1});
  CATCH_REQUIRE(model->records[0].positionIds == std::vector<fl::LongType>{0, 1, 2, 0});

  std::vector<RequestOutput> second = scheduler.step();
  CATCH_REQUIRE(second.size() == 2);
  CATCH_REQUIRE(second[0].tokenIds == std::vector<fl::LongType>{5});
  CATCH_REQUIRE(second[0].finished);
  CATCH_REQUIRE(second[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(second[1].tokenIds == std::vector<fl::LongType>{7});
  CATCH_REQUIRE(second[1].finished);
  CATCH_REQUIRE(second[1].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(model->records[1].queryLengths == std::vector<int>{1, 1});
  CATCH_REQUIRE(model->records[1].keyLengths == std::vector<int>{4, 2});
  CATCH_REQUIRE(model->records[1].positionIds == std::vector<fl::LongType>{3, 1});
  CATCH_REQUIRE_FALSE(scheduler.hasUnfinishedRequests());
  CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == NumBlocks);
}

CATCH_TEST_CASE("SchedulerV2 chunks prefill without sampling early", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>();
  SchedulerV2 scheduler(model, 2);
  scheduler.addRequest(request("a", {1, 2, 3}, 1));
  scheduler.addRequest(request("b", {5}, 1));

  CATCH_REQUIRE(scheduler.step().empty());
  CATCH_REQUIRE(model->records[0].queryLengths == std::vector<int>{2});
  CATCH_REQUIRE(model->records[0].positionIds == std::vector<fl::LongType>{0, 1});

  std::vector<RequestOutput> outputs = scheduler.step();
  CATCH_REQUIRE(outputs.size() == 2);
  CATCH_REQUIRE(outputs[0].tokenIds == std::vector<fl::LongType>{4});
  CATCH_REQUIRE(outputs[1].tokenIds == std::vector<fl::LongType>{6});
  CATCH_REQUIRE(outputs[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(outputs[1].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(model->records[1].queryLengths == std::vector<int>{1, 1});
  CATCH_REQUIRE(model->records[1].keyLengths == std::vector<int>{3, 1});
}

CATCH_TEST_CASE("SchedulerV2 batches mixed sampling parameters", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>();
  SchedulerV2 scheduler(model, 3);

  GenerationConfig greedy = config(1);
  greedy.temperature = 0.0f;
  greedy.topK = 0;
  GenerationConfig topK = config(1);
  topK.topK = 1;
  GenerationConfig topP = config(1);
  topP.topK = 0;
  topP.topP = 0.1f;

  scheduler.addRequest(std::make_shared<Request>("greedy", std::vector<fl::LongType>{1}, greedy));
  scheduler.addRequest(std::make_shared<Request>("top-k", std::vector<fl::LongType>{5}, topK));
  scheduler.addRequest(std::make_shared<Request>("top-p", std::vector<fl::LongType>{7}, topP));

  std::vector<RequestOutput> outputs = scheduler.step();
  CATCH_REQUIRE(outputs.size() == 3);
  CATCH_REQUIRE(outputs[0].tokenIds == std::vector<fl::LongType>{2});
  CATCH_REQUIRE(outputs[1].tokenIds == std::vector<fl::LongType>{6});
  CATCH_REQUIRE(outputs[2].tokenIds == std::vector<fl::LongType>{8});
  for (const RequestOutput &output : outputs) {
    CATCH_REQUIRE(output.finished);
    CATCH_REQUIRE(output.finishReason == RequestFinishReason::kLength);
  }
  CATCH_REQUIRE(model->records.size() == 1);
  CATCH_REQUIRE(model->records[0].queryLengths == std::vector<int>{1, 1, 1});
}

CATCH_TEST_CASE("SchedulerV2 delivers stop and cancellation", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>();
  SchedulerV2 scheduler(model, 4);
  scheduler.addRequest(request("stop", {8}, 4));
  scheduler.addRequest(request("cancel", {1, 2}, 4));
  scheduler.abortRequest("cancel");
  scheduler.abortRequest("missing");

  std::vector<RequestOutput> outputs = scheduler.step();
  CATCH_REQUIRE(outputs.size() == 2);
  CATCH_REQUIRE(outputs[0].requestId == "cancel");
  CATCH_REQUIRE(outputs[0].finished);
  CATCH_REQUIRE(outputs[0].finishReason == RequestFinishReason::kCancelled);
  CATCH_REQUIRE(outputs[1].requestId == "stop");
  CATCH_REQUIRE(outputs[1].finished);
  CATCH_REQUIRE(outputs[1].tokenIds.empty());
  CATCH_REQUIRE(outputs[1].finishReason == RequestFinishReason::kStop);
  CATCH_REQUIRE_FALSE(scheduler.hasUnfinishedRequests());
  CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == NumBlocks);
}

CATCH_TEST_CASE("SchedulerV2 preempts and recomputes under KV pressure", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>(2);
  SchedulerV2 scheduler(model, 6);
  scheduler.addRequest(request("a", {1, 2, 3}, 3));
  scheduler.addRequest(request("b", {10, 11, 12}, 3));
  scheduler.addRequest(request("c", {6}, 1));

  CATCH_REQUIRE(scheduler.step().size() == 2);
  CATCH_REQUIRE(scheduler.step().size() == 2);
  CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == 0);

  std::vector<RequestOutput> third = scheduler.step();
  CATCH_REQUIRE(third.size() == 1);
  CATCH_REQUIRE(third[0].requestId == "a");
  CATCH_REQUIRE(third[0].finished);
  CATCH_REQUIRE(third[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(scheduler.hasUnfinishedRequests());

  std::vector<RequestOutput> fourth = scheduler.step();
  CATCH_REQUIRE(fourth.size() == 1);
  CATCH_REQUIRE(fourth[0].requestId == "b");
  CATCH_REQUIRE(fourth[0].tokenIds == std::vector<fl::LongType>{15});
  CATCH_REQUIRE(fourth[0].finished);
  CATCH_REQUIRE(fourth[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(model->records.back().queryLengths == std::vector<int>{5});
  CATCH_REQUIRE(model->records.back().positionIds ==
                std::vector<fl::LongType>{0, 1, 2, 3, 4});
  CATCH_REQUIRE(scheduler.hasUnfinishedRequests());

  std::vector<RequestOutput> fifth = scheduler.step();
  CATCH_REQUIRE(fifth.size() == 1);
  CATCH_REQUIRE(fifth[0].requestId == "c");
  CATCH_REQUIRE(fifth[0].finished);
  CATCH_REQUIRE(fifth[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE_FALSE(scheduler.hasUnfinishedRequests());
  CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == 2);
}

CATCH_TEST_CASE("SchedulerV2 self-preempts when no other victim is available", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>(3);
  SchedulerV2 scheduler(model, 6);
  scheduler.addRequest(request("a", {1, 2, 3, 4, 5}, 2));
  std::shared_ptr<Request> requestB = request("b", {10, 11, 12, 13, 14}, 1);
  scheduler.addRequest(requestB);

  std::vector<RequestOutput> first = scheduler.step();
  CATCH_REQUIRE(first.size() == 1);
  CATCH_REQUIRE(first[0].requestId == "a");
  CATCH_REQUIRE(requestB->getContextLength() == 1);
  CATCH_REQUIRE(requestB->getBlockIds().size() == 1);
  CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == 0);

  std::vector<RequestOutput> second = scheduler.step();
  CATCH_REQUIRE(second.size() == 1);
  CATCH_REQUIRE(second[0].requestId == "a");
  CATCH_REQUIRE(second[0].finished);
  CATCH_REQUIRE(second[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(requestB->getNumComputedTokens() == 0);
  CATCH_REQUIRE(requestB->getContextLength() == 0);
  CATCH_REQUIRE(requestB->getBlockIds().empty());
  CATCH_REQUIRE(scheduler.hasUnfinishedRequests());

  std::vector<RequestOutput> third = scheduler.step();
  CATCH_REQUIRE(third.size() == 1);
  CATCH_REQUIRE(third[0].requestId == "b");
  CATCH_REQUIRE(third[0].tokenIds == std::vector<fl::LongType>{15});
  CATCH_REQUIRE(third[0].finished);
  CATCH_REQUIRE(third[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(model->records.back().queryLengths == std::vector<int>{5});
  CATCH_REQUIRE(model->records.back().positionIds ==
                std::vector<fl::LongType>{0, 1, 2, 3, 4});
  CATCH_REQUIRE_FALSE(scheduler.hasUnfinishedRequests());
  CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == 3);
}

CATCH_TEST_CASE("SchedulerV2 does not admit waiting requests after preemption", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>(3);
  SchedulerV2 scheduler(model, 8);
  std::shared_ptr<Request> requestA = request("a", {1, 2, 3}, 3);
  std::shared_ptr<Request> requestB = request("b", {10, 11, 12, 13, 14}, 10);
  std::shared_ptr<Request> requestC = request("c", {6}, 1);
  scheduler.addRequest(requestA);
  scheduler.addRequest(requestB);
  scheduler.addRequest(requestC);

  CATCH_REQUIRE(requestA->getStatus() == RequestStatus::kWaiting);
  CATCH_REQUIRE(requestB->getStatus() == RequestStatus::kWaiting);
  CATCH_REQUIRE(requestC->getStatus() == RequestStatus::kWaiting);

  CATCH_REQUIRE(scheduler.step().size() == 2);
  CATCH_REQUIRE(requestA->getStatus() == RequestStatus::kRunning);
  CATCH_REQUIRE(requestB->getStatus() == RequestStatus::kRunning);
  CATCH_REQUIRE(requestC->getStatus() == RequestStatus::kWaiting);
  CATCH_REQUIRE(scheduler.step().size() == 2);
  CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == 0);

  std::vector<RequestOutput> third = scheduler.step();
  CATCH_REQUIRE(third.size() == 1);
  CATCH_REQUIRE(third[0].requestId == "a");
  CATCH_REQUIRE(third[0].finished);
  CATCH_REQUIRE(third[0].finishReason == RequestFinishReason::kLength);
  CATCH_REQUIRE(requestA->getStatus() == RequestStatus::kFinished);
  CATCH_REQUIRE(requestB->getStatus() == RequestStatus::kPreempted);
  CATCH_REQUIRE(requestC->getStatus() == RequestStatus::kWaiting);
  CATCH_REQUIRE(model->records.back().queryLengths == std::vector<int>{1});
  CATCH_REQUIRE(requestB->getContextLength() == 0);
  CATCH_REQUIRE(requestB->getBlockIds().empty());
  CATCH_REQUIRE(requestC->getContextLength() == 0);
  CATCH_REQUIRE(requestC->getBlockIds().empty());
}

CATCH_TEST_CASE("SchedulerV2 rejects requests exceeding KV capacity", "[libllm][scheduler_v2]") {
  auto requireCapacityError = [](std::shared_ptr<FakeModel> model, int promptLength) {
    int numFreeBlocks = model->cache()->getNumFreeBlocks();
    SchedulerV2 scheduler(model, promptLength);
    scheduler.addRequest(request(
        "oversized",
        std::vector<fl::LongType>(promptLength, 1),
        1));

    std::vector<RequestOutput> outputs = scheduler.step();
    CATCH_REQUIRE(outputs.size() == 1);
    CATCH_REQUIRE(outputs[0].requestId == "oversized");
    CATCH_REQUIRE(outputs[0].finished);
    CATCH_REQUIRE(outputs[0].finishReason == RequestFinishReason::kError);
    CATCH_REQUIRE(outputs[0].errorMessage ==
                  "request exceeds the KV cache context capacity");
    CATCH_REQUIRE(model->records.empty());
    CATCH_REQUIRE(model->cache()->getNumFreeBlocks() == numFreeBlocks);
    CATCH_REQUIRE_FALSE(scheduler.hasUnfinishedRequests());
  };

  CATCH_SECTION("model context length") {
    requireCapacityError(std::make_shared<FakeModel>(), 33);
  }

  CATCH_SECTION("total KV blocks") {
    requireCapacityError(std::make_shared<FakeModel>(2), 9);
  }
}

CATCH_TEST_CASE("SchedulerV2 validates requests", "[libllm][scheduler_v2]") {
  auto model = std::make_shared<FakeModel>();
  SchedulerV2 scheduler(model, 4);
  scheduler.addRequest(request("same", {1}, 1));
  CATCH_REQUIRE_THROWS(scheduler.addRequest(request("same", {2}, 1)));

  GenerationConfig invalid = config(1);
  invalid.topK = -2;
  CATCH_REQUIRE_THROWS(
      scheduler.addRequest(std::make_shared<Request>("invalid", std::vector<fl::LongType>{1}, invalid)));
}

}  // namespace libllm
