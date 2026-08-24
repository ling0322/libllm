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

#include <stddef.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "flint/device.h"
#include "flint/operators.h"
#include "libllm/engine.h"
#include "libllm/engine_config.h"
#include "libllm/model_for_generation.h"
#include "libllm/request.h"
#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/zip_file.h"

namespace libllm {
namespace api {

thread_local int gErrorCode = 0;
thread_local char gErrorMessage[512] = "";

void llmSetErrorMessage(const std::string &message) {
  std::string what = message;
  if (what.size() >= sizeof(gErrorMessage)) {
    what.erase(what.begin() + sizeof(gErrorMessage) - 4, what.end());
    what += "...";
  }
  snprintf(gErrorMessage, sizeof(gErrorMessage), "%s", what.c_str());
}

}  // namespace api
}  // namespace libllm

struct llm_engine_impl_t {
  std::shared_ptr<lut::ZipFile> package;
  std::shared_ptr<libllm::ModelForGeneration> model;
  std::unique_ptr<libllm::Engine> engine;
  llm_stream_callback_t callback = nullptr;
  void *userData = nullptr;
};

namespace {

std::once_flag gLlmInitOnce;

void clearError() {
  libllm::api::gErrorCode = 0;
  libllm::api::llmSetErrorMessage("");
}

int32_t setError(int32_t code, const std::string &message) {
  libllm::api::gErrorCode = code;
  libllm::api::llmSetErrorMessage(message);
  return code;
}

bool isValidStringView(llm_string_view_t value) {
  return value.size >= 0 && (value.data != nullptr || value.size == 0);
}

std::string copyString(llm_string_view_t value) {
  return value.size == 0
      ? std::string()
      : std::string(value.data, static_cast<size_t>(value.size));
}

fl::Device getDevice(llm_device_type_t device) {
  switch (device) {
    case LLM_DEVICE_AUTO:
      return fl::Device::isCudaAvailable() ? fl::Device::getCuda() : fl::Device::getCpu();
    case LLM_DEVICE_CPU:
      return fl::Device::getCpu();
    case LLM_DEVICE_CUDA:
      return fl::Device::getCuda();
  }
  throw lut::AbortedError("invalid engine device");
}

llm_finish_reason_t getFinishReason(libllm::RequestFinishReason reason) {
  switch (reason) {
    case libllm::RequestFinishReason::kNone:
      return LLM_FINISH_REASON_NONE;
    case libllm::RequestFinishReason::kStop:
      return LLM_FINISH_REASON_STOP;
    case libllm::RequestFinishReason::kLength:
      return LLM_FINISH_REASON_LENGTH;
    case libllm::RequestFinishReason::kCancelled:
      return LLM_FINISH_REASON_CANCELLED;
    case libllm::RequestFinishReason::kError:
      return LLM_FINISH_REASON_ERROR;
  }
  return LLM_FINISH_REASON_ERROR;
}

llm_string_view_t getStringView(const std::string &value) {
  return {value.empty() ? nullptr : value.data(), static_cast<int64_t>(value.size())};
}

void emitOutputs(
    llm_engine_impl_t *engine,
    const std::vector<libllm::RequestOutput> &outputs) {
  std::vector<llm_request_output_t> views;
  views.reserve(outputs.size());
  for (const libllm::RequestOutput &output : outputs) {
    llm_request_output_t view{};
    view.request_id = getStringView(output.requestId);
    view.token_ids = output.tokenIds.empty() ? nullptr : output.tokenIds.data();
    view.num_token_ids = static_cast<int64_t>(output.tokenIds.size());
    view.text = getStringView(output.text);
    view.finished = output.finished ? 1 : 0;
    view.finish_reason = getFinishReason(output.finishReason);
    view.error_message = getStringView(output.errorMessage);
    views.push_back(view);
  }

  llm_request_outputs_t batch{
      views.empty() ? nullptr : views.data(),
      static_cast<int64_t>(views.size())};
  engine->callback(&batch, engine->userData);
}

bool hasCurrentStructSize(int64_t actual, size_t expected) {
  return actual >= 0 && static_cast<uint64_t>(actual) >= expected;
}

}  // namespace

void llm_init() {
  try {
    std::call_once(gLlmInitOnce, []() {
      lut::setLogLevel(lut::LogSeverity::kINFO);
      fl::initOperators();
    });
    clearError();
  } catch (const std::exception &error) {
    setError(LLM_ERROR_ABORTED, error.what());
  }
}

void llm_engine_options_init(llm_engine_options_t *options) {
  if (!options) return;

  *options = {};
  options->struct_size = sizeof(*options);
  options->device = LLM_DEVICE_AUTO;
  options->max_num_batched_tokens = 2048;
  options->kv_cache_block_size = 256;
  options->kv_cache_memory_utilization = 0.9f;
}

void llm_generation_config_init(llm_generation_config_t *config) {
  if (!config) return;

  *config = {};
  config->struct_size = sizeof(*config);
  config->top_k = 0;
  config->top_p = 1.0f;
  config->temperature = 1.0f;
  config->max_tokens = std::numeric_limits<int32_t>::max();
}

void llm_request_init(llm_request_t *request) {
  if (!request) return;

  *request = {};
  request->struct_size = sizeof(*request);
  llm_generation_config_init(&request->generation_config);
}

int32_t llm_get_last_error_code() {
  return libllm::api::gErrorCode;
}

const char *llm_get_last_error_message() {
  return libllm::api::gErrorMessage;
}

int32_t llm_engine_init(llm_engine_t *engine) {
  if (!engine) return setError(LLM_ERROR_INVALID_ARG, "engine is null");
  if (*engine) return setError(LLM_ERROR_INVALID_ARG, "engine is already initialized");

  try {
    *engine = new llm_engine_impl_t();
    clearError();
    return 0;
  } catch (const std::exception &error) {
    return setError(LLM_ERROR_ABORTED, error.what());
  }
}

int32_t llm_engine_load(
    llm_engine_t *engine,
    const llm_engine_options_t *options,
    llm_stream_callback_t callback,
    void *userData) {
  if (!engine || !*engine) {
    return setError(LLM_ERROR_INVALID_ARG, "engine is not initialized");
  }
  if (!options || !hasCurrentStructSize(options->struct_size, sizeof(*options))) {
    return setError(LLM_ERROR_INVALID_ARG, "invalid engine options struct_size");
  }
  if (!callback) return setError(LLM_ERROR_INVALID_ARG, "callback is null");
  if ((*engine)->engine) return setError(LLM_ERROR_INVALID_ARG, "engine is already loaded");
  if (!isValidStringView(options->model_path) || options->model_path.size == 0) {
    return setError(LLM_ERROR_INVALID_ARG, "model_path is empty");
  }
  if (options->max_num_batched_tokens <= 0) {
    return setError(LLM_ERROR_INVALID_ARG, "max_num_batched_tokens must be positive");
  }
  if (options->kv_cache_block_size <= 0 ||
      (options->kv_cache_block_size & (options->kv_cache_block_size - 1)) != 0) {
    return setError(LLM_ERROR_INVALID_ARG, "kv_cache_block_size must be a power of two");
  }
  if (!std::isfinite(options->kv_cache_memory_utilization) ||
      options->kv_cache_memory_utilization <= 0.0f ||
      options->kv_cache_memory_utilization > 1.0f) {
    return setError(LLM_ERROR_INVALID_ARG, "kv_cache_memory_utilization must be in (0, 1]");
  }
  llm_engine_impl_t *impl = *engine;
  try {
    llm_init();
    if (llm_get_last_error_code() != 0) {
      throw lut::AbortedError(llm_get_last_error_message());
    }
    std::string modelPath = copyString(options->model_path);
    fl::Device device = getDevice(options->device);
    libllm::EngineConfig config;
    config.maxNumBatchedTokens = options->max_num_batched_tokens;
    config.kvCacheBlockSize = options->kv_cache_block_size;
    config.kvCacheMemoryUtilization = options->kv_cache_memory_utilization;

    impl->package = lut::ZipFile::fromFile(modelPath);
    impl->model = libllm::ModelForGeneration::fromPackage(
        device, impl->package.get(), config);
    impl->callback = callback;
    impl->userData = userData;
    impl->engine = std::make_unique<libllm::Engine>(
        impl->model,
        config.maxNumBatchedTokens,
        [impl](const std::vector<libllm::RequestOutput> &outputs) {
          emitOutputs(impl, outputs);
        });
    clearError();
    return 0;
  } catch (const std::exception &error) {
    impl->engine.reset();
    impl->model.reset();
    impl->package.reset();
    impl->callback = nullptr;
    impl->userData = nullptr;
    return setError(LLM_ERROR_ABORTED, error.what());
  }
}

int32_t llm_engine_destroy(llm_engine_t *engine) {
  if (!engine || !*engine) {
    clearError();
    return 0;
  }

  llm_engine_impl_t *impl = *engine;
  try {
    if (impl->engine) impl->engine->shutdown();
    delete impl;
    *engine = nullptr;
    clearError();
    return 0;
  } catch (const std::exception &error) {
    delete impl;
    *engine = nullptr;
    return setError(LLM_ERROR_ABORTED, error.what());
  }
}

int32_t llm_engine_add_request(
    llm_engine_t *engine,
    const llm_request_t *request) {
  if (!engine || !*engine || !(*engine)->engine) {
    return setError(LLM_ERROR_INVALID_ARG, "engine is not loaded");
  }
  if (!request || !hasCurrentStructSize(request->struct_size, sizeof(*request))) {
    return setError(LLM_ERROR_INVALID_ARG, "invalid request struct_size");
  }
  if (!hasCurrentStructSize(
          request->generation_config.struct_size,
          sizeof(request->generation_config))) {
    return setError(LLM_ERROR_INVALID_ARG, "invalid generation_config struct_size");
  }
  if (!isValidStringView(request->request_id) || request->request_id.size == 0) {
    return setError(LLM_ERROR_INVALID_ARG, "request_id is empty");
  }
  bool hasInputIds = request->input_ids && request->num_input_ids > 0;
  bool hasMessages = request->messages && request->num_messages > 0;
  if (hasInputIds == hasMessages) {
    return setError(LLM_ERROR_INVALID_ARG, "exactly one request input form is required");
  }
  if (request->num_input_ids < 0 || request->num_messages < 0 ||
      request->num_input_ids > std::numeric_limits<int>::max() ||
      request->num_messages > std::numeric_limits<int>::max()) {
    return setError(LLM_ERROR_INVALID_ARG, "request input is too large");
  }
  const llm_generation_config_t &generation = request->generation_config;
  if (!std::isfinite(generation.temperature) || generation.temperature < 0.0f) {
    return setError(LLM_ERROR_INVALID_ARG, "temperature must be finite and not negative");
  }
  if (generation.top_k < -1) {
    return setError(LLM_ERROR_INVALID_ARG, "top_k must be zero, -1, or positive");
  }
  if (generation.top_k > (*engine)->model->getVocab()->getVocabSize()) {
    return setError(LLM_ERROR_INVALID_ARG, "top_k exceeds the model vocabulary size");
  }
  if (!std::isfinite(generation.top_p) ||
      generation.top_p <= 0.0f || generation.top_p > 1.0f) {
    return setError(LLM_ERROR_INVALID_ARG, "top_p must be in (0, 1]");
  }
  if (generation.max_tokens < 0) {
    return setError(LLM_ERROR_INVALID_ARG, "max_tokens must not be negative");
  }
  if (hasInputIds) {
    int vocabSize = (*engine)->model->getVocab()->getVocabSize();
    for (int64_t i = 0; i < request->num_input_ids; ++i) {
      if (request->input_ids[i] < 0 || request->input_ids[i] >= vocabSize) {
        return setError(LLM_ERROR_INVALID_ARG, "input_ids contains an out-of-range token");
      }
    }
  } else {
    for (int64_t i = 0; i < request->num_messages; ++i) {
      if (!isValidStringView(request->messages[i].role) ||
          request->messages[i].role.size == 0 ||
          !isValidStringView(request->messages[i].content)) {
        return setError(LLM_ERROR_INVALID_ARG, "messages contains an invalid role or content");
      }
    }
  }

  try {
    libllm::GenerationConfig config;
    config.topK = generation.top_k;
    config.topP = generation.top_p;
    config.temperature = generation.temperature;
    config.maxTokens = generation.max_tokens;

    std::vector<fl::LongType> inputIds;
    if (hasInputIds) {
      inputIds.assign(
          request->input_ids,
          request->input_ids + request->num_input_ids);
    } else {
      std::vector<libllm::Message> messages;
      messages.reserve(static_cast<size_t>(request->num_messages));
      for (int64_t i = 0; i < request->num_messages; ++i) {
        messages.push_back({
            copyString(request->messages[i].role),
            copyString(request->messages[i].content)});
      }
      libllm::Prompt prompt = (*engine)->model->buildPrompt(messages);
      inputIds = (*engine)->model->encodePrompt(prompt);
    }
    auto typedRequest = std::make_shared<libllm::Request>(
        copyString(request->request_id),
        std::move(inputIds),
        config);
    (*engine)->engine->addRequest(std::move(typedRequest));
    clearError();
    return 0;
  } catch (const std::exception &error) {
    return setError(LLM_ERROR_ABORTED, error.what());
  }
}

int32_t llm_engine_abort_request(
    llm_engine_t *engine,
    llm_string_view_t requestId) {
  if (!engine || !*engine || !(*engine)->engine) {
    return setError(LLM_ERROR_INVALID_ARG, "engine is not loaded");
  }
  if (!isValidStringView(requestId) || requestId.size == 0) {
    return setError(LLM_ERROR_INVALID_ARG, "request_id is empty");
  }

  try {
    (*engine)->engine->abortRequest(copyString(requestId));
    clearError();
    return 0;
  } catch (const std::exception &error) {
    return setError(LLM_ERROR_ABORTED, error.what());
  }
}
