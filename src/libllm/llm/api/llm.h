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

#include <stdint.h>

#define LLMAPI

typedef int32_t LLM_RESULT;
typedef int32_t LLM_BOOL;

#define LLM_TRUE 1
#define LLM_FALSE 0
#define LLM_OK 0

typedef struct llm_model_t llm_model_t;
typedef struct llm_compl_opt_t llm_compl_opt_t;
typedef struct llm_compl_t llm_compl_t;
typedef struct llm_chunk_t llm_chunk_t;

#ifdef __cplusplus
extern "C" {
#endif

LLMAPI LLM_RESULT llm_init();
LLMAPI void llm_destroy();

LLMAPI llm_model_t *llm_model_init(const char *ini_path);
LLMAPI void llm_model_destroy(llm_model_t *m);
LLMAPI const char *llm_model_get_name(llm_model_t *m);

LLMAPI llm_compl_opt_t *llm_compl_opt_init();
LLMAPI void llm_compl_opt_destroy(llm_compl_opt_t *o);

LLMAPI LLM_RESULT llm_compl_opt_set_top_p(llm_compl_opt_t *o, float topp);
LLMAPI LLM_RESULT llm_compl_opt_set_temperature(llm_compl_opt_t *o, float temperature);
LLMAPI LLM_RESULT llm_compl_opt_set_prompt(llm_compl_opt_t *o, const char *prompt);
LLMAPI LLM_RESULT llm_compl_opt_set_top_k(llm_compl_opt_t *o, int32_t topk);
LLMAPI LLM_RESULT llm_compl_opt_set_model(llm_compl_opt_t *o, llm_model_t *model);

LLMAPI llm_compl_t *llm_compl_init(llm_compl_opt_t *o);
LLMAPI void llm_compl_destroy(llm_compl_t *c);
LLMAPI LLM_BOOL llm_compl_stopped(llm_compl_t *c);
LLMAPI llm_chunk_t *llm_compl_next_chunk(llm_compl_t *c);

LLMAPI const char *llm_chunk_get_text(llm_chunk_t *c);
LLMAPI void llm_chunk_destroy(llm_chunk_t *c);

#ifdef __cplusplus
}  // extern "C"

#include <memory>
#include <stdexcept>


namespace llm {

class Model;
class Completion;

// configuration for LLM completion task.
class CompletionConfig {
 public:
  CompletionConfig() :
      _topP(0.8),
      _topK(50),
      _temperature(1.0) {}

  // setters for the config.
  void setTopP(float topP) { _topP = topP; }
  void setTopK(int topK) { _topK = topK; }
  void setTemperature(float temperature) { _temperature = temperature; }

  // getters for the config.
  float getTopP() const { return _topP; }
  int getTopK() const { return _topK; }
  int getTemperature() const { return _temperature; }

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

  bool stopped();
  Chunk nextChunk();

 private:
  std::shared_ptr<llm_compl_t> _completion;
  bool _stopped;
  Chunk _chunk;
};

/// @brief Stores an instance of LLM Model.
class Model {
 public:
  /// @brief Create an instance of Model from the config file path;
  /// @param configFile config file of the model.
  /// @return A shared pointer of the Model instance.
  static std::shared_ptr<Model> create(const std::string &configFile);

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

inline void init() {
  LLM_RESULT r = llm_init();
  if (r != LLM_OK) {
    throw std::runtime_error("llmpp initialization failed.");
  }
}

inline void destroy() {
  llm_destroy();
}

inline std::shared_ptr<llm_compl_opt_t> CompletionConfig::getInternalOption() {
  std::shared_ptr<llm_compl_opt_t> option(
      llm_compl_opt_init(),
      llm_compl_opt_destroy);
  if (!option) throw std::runtime_error("create option failed.");

  if (LLM_OK != llm_compl_opt_set_top_p(option.get(), getTopP()))
    throw std::runtime_error("invalid top-p.");

  if (LLM_OK != llm_compl_opt_set_top_k(option.get(), getTopK()))
    throw std::runtime_error("invalid top-k.");

  if (LLM_OK != llm_compl_opt_set_temperature(option.get(), getTemperature()))
    throw std::runtime_error("invalid temperature.");
  
  return option;
}

inline bool Completion::stopped() {
  return llm_compl_stopped(_completion.get()) == LLM_TRUE;
}

inline Chunk Completion::nextChunk() {
  std::shared_ptr<llm_chunk_t> chunk(llm_compl_next_chunk(_completion.get()),
                                       llm_chunk_destroy);
  if (!chunk) throw std::runtime_error("failed to get chunk.");

  const char *text = llm_chunk_get_text(chunk.get());
  if (!text) throw std::runtime_error("failed to get text.");

  Chunk c;
  c._text = text;
  return c;
}

inline std::shared_ptr<Model> Model::create(const std::string &iniPath) {
  llm_model_t *model_ptr = llm_model_init(iniPath.c_str());
  if (!model_ptr) throw std::runtime_error("create model failed.");

  std::shared_ptr<Model> model{new Model()};
  model->_model = {model_ptr, llm_model_destroy};
  return model;
}

inline const char *Model::getName() {
  return llm_model_get_name(_model.get());
}

inline Completion Model::complete(const std::string &prompt, CompletionConfig config) {
  std::shared_ptr<llm_compl_opt_t> option = config.getInternalOption();

  if (LLM_OK != llm_compl_opt_set_prompt(option.get(), prompt.c_str()))
    throw std::runtime_error("invalid prompt.");

  if (LLM_OK != llm_compl_opt_set_model(option.get(), _model.get()))
    throw std::runtime_error("invalid model.");

  std::shared_ptr<llm_compl_t> completion(
      llm_compl_init(option.get()),
      llm_compl_destroy);

  Completion c;
  c._completion = completion;
  return c;
}

}  // namespace llm

#endif  // __cplusplus
