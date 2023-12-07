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

// C++ wrapper for libllm C API.

#pragma once

#include <memory>
#include <stdexcept>
#include "llm.h"

namespace llm {

class Model;
class Completion;

// configuration for LLM completion task.
class CompletionConfig {
 public:
  CompletionConfig() :
      _topP(0.8f),
      _topK(50),
      _temperature(1.0f) {}

  // setters for the config.
  void setTopP(float topP) { _topP = topP; }
  void setTopK(int topK) { _topK = topK; }
  void setTemperature(float temperature) { _temperature = temperature; }

  // getters for the config.
  float getTopP() const { return _topP; }
  int getTopK() const { return _topK; }
  float getTemperature() const { return _temperature; }

  // gets the instance of C-API llm_compl_opt_t object.
  std::shared_ptr<llm_compl_opt_t> getInternalOption();

 private:
  float _topP;
  int _topK;
  float _temperature;
};

class Chunk {
 public:
  friend class Completion;

  std::string getText() const { return _text; }

 private:
  std::string _text;
};

class Completion {
 public:
  friend class Model;

  Completion() : _stopped(true) {}

  /// @brief If completion is ongoing (active) returns true, if stopped returns false.
  /// @return If completion is active.
  bool isActive();

  Chunk nextChunk();

 private:
  std::shared_ptr<llm_compl_t> _completion;
  bool _stopped;
  Chunk _chunk;
};

enum class DeviceType {
  CPU = LIBLLM_DEVICE_CPU,
  CUDA = LIBLLM_DEVICE_CUDA,
  AUTO = LIBLLM_DEVICE_AUTO
};

/// @brief Stores an instance of LLM Model.
class Model {
 public:
  /// @brief Create an instance of Model from the config file path;
  /// @param configFile config file of the model.
  /// @param device device of the model storage and computation device. Use DeviceType::AUTO to
  /// let libllm determine the best one.
  /// @return A shared pointer of the Model instance.
  static std::shared_ptr<Model> create(const std::string &configFile,
                                       DeviceType device = DeviceType::AUTO);

  /// @brief Get the name of model, for example, "llama".
  /// @return name of the model.
  const char *getName();

  /// @brief Complete the given `prompt` with LLM.
  /// @param prompt The prompt to complete.
  /// @param config The config for completion.
  /// @return A `Completion` object.
  Completion complete(const std::string &prompt,
                      CompletionConfig config = CompletionConfig());

 private:
  std::shared_ptr<llm_model_t> _model;

  Model() = default;
};

// -- Implementation of libLLM C++ API (wrapper for C api) ----------------------------------------

namespace internal {

inline void throwLastError() {
  std::string lastError = llm_get_last_error_message();
  throw std::runtime_error(lastError);
}

}  // namespace internal

inline void init() {
  LIBLLM_STATUS status = llm_init();
  if (status != LIBLLM_OK) internal::throwLastError();
}

inline void destroy() {
  llm_destroy();
}

inline std::shared_ptr<llm_compl_opt_t> CompletionConfig::getInternalOption() {
  std::shared_ptr<llm_compl_opt_t> option(
      llm_compl_opt_init(),
      llm_compl_opt_destroy);
  if (!option) throw std::runtime_error("create option failed.");

  if (LIBLLM_OK != llm_compl_opt_set_top_p(option.get(), getTopP()))
    internal::throwLastError();

  if (LIBLLM_OK != llm_compl_opt_set_top_k(option.get(), getTopK()))
    internal::throwLastError();

  if (LIBLLM_OK != llm_compl_opt_set_temperature(option.get(), getTemperature()))
    internal::throwLastError();
  
  return option;
}

inline bool Completion::isActive() {
  return llm_compl_is_active(_completion.get()) == LIBLLM_TRUE;
}

inline Chunk Completion::nextChunk() {
  std::shared_ptr<llm_chunk_t> chunk(llm_compl_next_chunk(_completion.get()),
                                       llm_chunk_destroy);
  if (!chunk) internal::throwLastError();

  const char *text = llm_chunk_get_text(chunk.get());
  if (!text) internal::throwLastError();

  Chunk c;
  c._text = text;
  return c;
}

inline std::shared_ptr<Model> Model::create(const std::string &iniPath, DeviceType device) {
  std::shared_ptr<llm_model_opt_t> modelOpt(
      llm_model_opt_init(iniPath.c_str()),
      llm_model_opt_destroy);
  if (LIBLLM_OK != llm_model_opt_set_device(modelOpt.get(), int(device))) 
    internal::throwLastError();

  llm_model_t *model_ptr = llm_model_init(modelOpt.get());
  if (!model_ptr) internal::throwLastError();

  std::shared_ptr<Model> model{new Model()};
  model->_model = {model_ptr, llm_model_destroy};
  return model;
}

inline const char *Model::getName() {
  return llm_model_get_name(_model.get());
}

inline Completion Model::complete(const std::string &prompt, CompletionConfig config) {
  std::shared_ptr<llm_compl_opt_t> option = config.getInternalOption();

  if (LIBLLM_OK != llm_compl_opt_set_prompt(option.get(), prompt.c_str()))
    internal::throwLastError();

  std::shared_ptr<llm_compl_t> completion(llm_model_complete(_model.get(), option.get()),
                                          llm_compl_destroy);
  if (!completion) internal::throwLastError();

  Completion c;
  c._completion = completion;
  return c;
}

}  // namespace llm