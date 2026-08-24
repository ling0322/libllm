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

#include "libllm/llm_v2.h"

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/operators.h"

namespace {

llm_string_view_t stringView(const std::string &value) {
  return {value.data(), static_cast<int64_t>(value.size())};
}

void discardOutputs(const llm_request_outputs_t *, void *) {
}

struct V2CallbackState {
  std::mutex mutex;
  std::condition_variable ready;
  std::string requestId;
  std::vector<int64_t> tokenIds;
  llm_finish_reason_t finishReason = LLM_FINISH_REASON_NONE;
  bool finished = false;
};

void collectOutputs(const llm_request_outputs_t *outputs, void *userData) {
  auto *state = static_cast<V2CallbackState *>(userData);
  std::lock_guard<std::mutex> lock(state->mutex);
  for (int64_t i = 0; i < outputs->size; ++i) {
    const llm_request_output_t &output = outputs->data[i];
    state->requestId = output.request_id.size == 0
        ? std::string()
        : std::string(
              output.request_id.data,
              static_cast<size_t>(output.request_id.size));
    if (output.num_token_ids == 0) {
      state->tokenIds.clear();
    } else {
      state->tokenIds.assign(
          output.token_ids,
          output.token_ids + output.num_token_ids);
    }
    if (output.finished) {
      state->finished = true;
      state->finishReason = output.finish_reason;
    }
  }
  state->ready.notify_all();
}

std::string findV2TestModel() {
  for (const std::string &path : {
           std::string("models/llama3.2-3b-instruct-fp16.llmpkg"),
           std::string("../models/llama3.2-3b-instruct-fp16.llmpkg")}) {
    if (std::ifstream(path, std::ios::binary).is_open()) return path;
  }
  return "";
}

}  // namespace

CATCH_TEST_CASE("V2 structs initialize to library defaults", "[libllm][llm_v2]") {
  llm_engine_options_t options;
  llm_engine_options_init(&options);
  CATCH_REQUIRE(options.struct_size == sizeof(options));
  CATCH_REQUIRE(options.model_path.data == nullptr);
  CATCH_REQUIRE(options.model_path.size == 0);
  CATCH_REQUIRE(options.device == LLM_DEVICE_AUTO);
  CATCH_REQUIRE(options.max_num_batched_tokens == 2048);
  CATCH_REQUIRE(options.kv_cache_block_size == 256);
  CATCH_REQUIRE(options.kv_cache_memory_utilization == 0.9f);

  llm_generation_config_t config;
  llm_generation_config_init(&config);
  CATCH_REQUIRE(config.struct_size == sizeof(config));
  CATCH_REQUIRE(config.top_k == 0);
  CATCH_REQUIRE(config.top_p == 1.0f);
  CATCH_REQUIRE(config.temperature == 1.0f);
  CATCH_REQUIRE(config.max_tokens == std::numeric_limits<int32_t>::max());

  llm_request_t request;
  llm_request_init(&request);
  CATCH_REQUIRE(request.struct_size == sizeof(request));
  CATCH_REQUIRE(request.request_id.data == nullptr);
  CATCH_REQUIRE(request.request_id.size == 0);
  CATCH_REQUIRE(request.input_ids == nullptr);
  CATCH_REQUIRE(request.num_input_ids == 0);
  CATCH_REQUIRE(request.messages == nullptr);
  CATCH_REQUIRE(request.num_messages == 0);
  CATCH_REQUIRE(request.generation_config.struct_size ==
                sizeof(request.generation_config));
  CATCH_REQUIRE(request.generation_config.top_k == 0);
  CATCH_REQUIRE(request.generation_config.top_p == 1.0f);
  CATCH_REQUIRE(request.generation_config.temperature == 1.0f);
  CATCH_REQUIRE(request.generation_config.max_tokens ==
                std::numeric_limits<int32_t>::max());

  llm_engine_options_init(nullptr);
  llm_generation_config_init(nullptr);
  llm_request_init(nullptr);
}

CATCH_TEST_CASE("V2 engine API validates lifecycle", "[libllm][llm_v2]") {
  llm_engine_t engine = nullptr;
  CATCH_REQUIRE(llm_engine_init(nullptr) == LLM_ERROR_INVALID_ARG);
  CATCH_REQUIRE(llm_get_last_error_code() == LLM_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(llm_get_last_error_message()).size() > 0);

  CATCH_REQUIRE(llm_engine_init(&engine) == 0);
  CATCH_REQUIRE(engine != nullptr);
  CATCH_REQUIRE(llm_get_last_error_code() == 0);
  CATCH_REQUIRE(llm_engine_init(&engine) == LLM_ERROR_INVALID_ARG);

  llm_request_t request;
  llm_request_init(&request);
  CATCH_REQUIRE(llm_engine_add_request(&engine, &request) == LLM_ERROR_INVALID_ARG);
  CATCH_REQUIRE(llm_engine_abort_request(&engine, {"missing", 7}) ==
                LLM_ERROR_INVALID_ARG);

  llm_engine_options_t options;
  llm_engine_options_init(&options);
  CATCH_REQUIRE(llm_engine_load(&engine, &options, discardOutputs, nullptr) ==
                LLM_ERROR_INVALID_ARG);

  std::string missingModel = "models/does-not-exist.llmpkg";
  options.model_path = stringView(missingModel);
  CATCH_REQUIRE(llm_engine_load(&engine, &options, discardOutputs, nullptr) ==
                LLM_ERROR_ABORTED);
  CATCH_REQUIRE(std::string(llm_get_last_error_message()).size() > 0);

  CATCH_REQUIRE(llm_engine_destroy(&engine) == 0);
  CATCH_REQUIRE(engine == nullptr);
  CATCH_REQUIRE(llm_get_last_error_code() == 0);
  CATCH_REQUIRE(llm_engine_destroy(&engine) == 0);
  CATCH_REQUIRE(llm_engine_destroy(nullptr) == 0);
}

CATCH_TEST_CASE("V2 engine API streams typed outputs", "[libllm][llm_v2]") {
  std::string modelPath = findV2TestModel();
  if (modelPath.empty()) CATCH_SKIP("test model is unavailable");

  llm_init();
  if (!fl::isOperatorsAvailable(fl::Device::kCuda)) CATCH_SKIP("CUDA is unavailable");

  V2CallbackState callbackState;
  llm_engine_t engine = nullptr;
  CATCH_REQUIRE(llm_engine_init(&engine) == 0);

  llm_engine_options_t options;
  llm_engine_options_init(&options);
  options.model_path = stringView(modelPath);
  options.device = LLM_DEVICE_CUDA;
  options.max_num_batched_tokens = 4;
  CATCH_REQUIRE(llm_engine_load(&engine, &options, collectOutputs, &callbackState) == 0);

  const std::string requestId = "v2-request";
  const std::string role = "user";
  const std::string content = "Hello";
  llm_message_t message{stringView(role), stringView(content)};
  llm_request_t request;
  llm_request_init(&request);
  request.request_id = stringView(requestId);
  request.messages = &message;
  request.num_messages = 1;
  request.generation_config.temperature = 0.0f;
  request.generation_config.max_tokens = 1;
  CATCH_REQUIRE(llm_engine_add_request(&engine, &request) == 0);

  {
    std::unique_lock<std::mutex> lock(callbackState.mutex);
    bool completed = callbackState.ready.wait_for(
        lock,
        std::chrono::seconds(10),
        [&]() { return callbackState.finished; });
    CATCH_REQUIRE(completed);
  }

  CATCH_REQUIRE(callbackState.requestId == requestId);
  CATCH_REQUIRE(callbackState.tokenIds.size() == 1);
  CATCH_REQUIRE(callbackState.finishReason == LLM_FINISH_REASON_LENGTH);
  CATCH_REQUIRE(llm_engine_abort_request(&engine, stringView("missing")) == 0);
  CATCH_REQUIRE(llm_engine_destroy(&engine) == 0);
  CATCH_REQUIRE(engine == nullptr);
}
