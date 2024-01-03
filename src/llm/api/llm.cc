#include "llm/api/llm.h"

#include <string.h>
#include <omp.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include "ly/ly.h"
#include "llm/api/model_factory.h"
#include "llm/chatglm/chatglm_model.h"
#include "llm/common/generator.h"
#include "llm/common/model_for_generation.h"
#include "lytok/lytok.h"
#include "lyutil/error.h"
#include "lyutil/ini_config.h"
#include "lyutil/log.h"

using libllm::Generator;
using libllm::GenerationConfig;
using libllm::ModelFactory;
using libllm::ModelForGeneration;
using libllm::chatglm::ChatGlmConfig;
using libllm::chatglm::ChatGlmModel;
using lytok::Tokenizer;
using lut::IniConfig;


struct llmModel_t {
  ly::Context ctx;
  std::shared_ptr<ModelForGeneration> model_for_generation;
  std::shared_ptr<Tokenizer> tokenizer;
  std::string configFile;
  int device;
};

struct llmCompletion_t {
  int top_k;
  float top_p;
  float temperature;
  std::vector<ly::LongType> prompt;
  std::weak_ptr<ModelForGeneration> model_for_generation;
  std::weak_ptr<Tokenizer> tokenizer;
  std::shared_ptr<Generator> generator;
};

struct llmChunk_t {
  std::string text;
};

struct llmPrompt_t {
  std::weak_ptr<Tokenizer> tokenizer;
  std::vector<ly::LongType> inputs;
};

namespace libllm {
namespace api {

thread_local int gErrorCode = static_cast<int>(lut::ErrorCode::OK);
thread_local char gErrorMessage[512] = "";
static std::atomic<bool> gInitialized{false};

void setErrorCodeAndMessage(const lut::Error &e) {
  gErrorCode = static_cast<int>(e.getCode());

  std::string what = e.what();
  if (what.size() >= sizeof(gErrorMessage)) {
    what.erase(what.begin() + sizeof(gErrorMessage) - 4, what.end());
    what += "...";
  }
  snprintf(gErrorMessage, sizeof(gErrorMessage), "%s", what.c_str());
}

llmStatus_t runAndCatch(std::function<void()> &&f) {
  try {
    f();
    return LLM_OK;
  } catch (const lut::Error &e) {
    setErrorCodeAndMessage(e);
    return static_cast<llmStatus_t>(e.getCode());
  }
}

template<typename T>
T runAndCatch(std::function<T()> &&c, T default_value) {
  try {
    return c();
  } catch (const lut::Error &e) {
    setErrorCodeAndMessage(e);
    return default_value;
  }
}

ly::Device getDeviceFromApi(int apiDevice) {
  switch (apiDevice) {
    case LLM_DEVICE_CPU:
      return ly::Device::getCpu();
    case LLM_DEVICE_CUDA:
      return ly::Device::getCuda();
    case LLM_DEVICE_AUTO:
      if (ly::Device::isCudaAvailable()) {
        return ly::Device::getCuda();
      } else {
        return ly::Device::getCpu();
      }
    default:
      throw lut::InvalidArgError("invalid device type");
  }
}

// -- api implementation ----------

llmStatus_t init() {
  if (!gInitialized.exchange(true)) {
    try {
      lut::setLogLevel(lut::LogSeverity::kINFO);
      ly::init();

      LOG(INFO) << "OMP max_threads = " << omp_get_max_threads();

      return LLM_OK;
    } catch (const lut::Error &e) {
      gInitialized = false;
      setErrorCodeAndMessage(e);
      return static_cast<llmStatus_t>(e.getCode());;
    }
  }

  return LLM_OK;
}

llmStatus_t destroy() {
  if (gInitialized.exchange(false)) {
    ly::destroy();
  }

  return LLM_OK;
}

const char *getLastErrorMessage() {
  return gErrorMessage;
}

llmModel_t *createModel() {
  llmModel_t *model = new llmModel_t();
  model->device = LLM_DEVICE_AUTO;
  return model;
}

llmStatus_t destroyModel(llmModel_t *model) {
  delete model;
  return LLM_OK;
}

llmStatus_t setModelConfig(llmModel_t *model, const char *filename) {
  return runAndCatch([model, filename](){
    if (!model) throw lut::InvalidArgError("model");
    if (!filename) throw lut::InvalidArgError("filename");

    model->configFile = filename;
    return LLM_OK;
  });
}

llmStatus_t setModelDevice(llmModel_t *model, int32_t device) {
  return runAndCatch([model, device](){
    if (!model) throw lut::InvalidArgError("model");

    model->device = device;
    return LLM_OK;
  });
}

llmStatus_t loadModel(llmModel_t *model) {
  return runAndCatch([model](){
    if (!model) throw lut::InvalidArgError("model");
    if (model->configFile.empty()) throw lut::InvalidArgError("model_config not set");

    std::unique_ptr<IniConfig> ini = IniConfig::read(model->configFile);

    model->ctx.setDevice(getDeviceFromApi(model->device));
    model->ctx.setFloatDType(ly::functional::getDefaultFloatType(model->ctx.getDevice()));
    model->tokenizer = Tokenizer::create(ini->getSection("tokenizer"));
    model->model_for_generation = ModelFactory::createModel(model->ctx, *ini);
  
    return LLM_OK;
  });
}

const char *getModelName(llmModel_t *model) {
  return runAndCatch<const char *>([model](){
    if (!model) throw lut::InvalidArgError("m");
    if (!model->model_for_generation) throw lut::InvalidArgError("model");

    return model->model_for_generation->getName();
  }, nullptr);
}

llmPrompt_t *createPrompt(llmModel_t *model) {
  return runAndCatch<llmPrompt_t *>([model](){
    if (!model) throw lut::InvalidArgError("model");
    if (!model->tokenizer) throw lut::InvalidArgError("model not initialized");

    llmPrompt_t *prompt = new llmPrompt_t();
    prompt->tokenizer = model->tokenizer;
    return prompt;
  }, nullptr);
}

llmStatus_t destroyPrompt(llmPrompt_t *prompt) {
  delete prompt;
  return LLM_OK;
}

llmStatus_t appendText(llmPrompt_t *prompt, const char *text) {
  return runAndCatch([prompt, text](){
    if (!prompt) throw lut::InvalidArgError("prompt");
    if (!text) throw lut::InvalidArgError("text");

    std::shared_ptr<lytok::Tokenizer> tokenizer = prompt->tokenizer.lock();
    if (!tokenizer) throw lut::AbortedError("tokenizer expired.");

    std::vector<int> inputIds = tokenizer->encode(text);
    for (int tokenId : inputIds) {
      prompt->inputs.push_back(tokenId);
      LOG(DEBUG) << "token " << tokenizer->getVocab()->getTokenString(tokenId) << " -> " << tokenId;
    }

    return LLM_OK;
  });
}

llmStatus_t appendControlToken(llmPrompt_t *prompt, const char *name) {
  return runAndCatch([prompt, name](){
    if (!prompt) throw lut::InvalidArgError("prompt");
    if (!name) throw lut::InvalidArgError("name");

    std::shared_ptr<lytok::Tokenizer> tokenizer = prompt->tokenizer.lock();
    if (!tokenizer) throw lut::AbortedError("tokenizer expired.");

    int tokenId = tokenizer->getVocab()->findControlToken(name);
    prompt->inputs.push_back(tokenId);
    LOG(DEBUG) << "control token " << name << " -> " << tokenId;

    return LLM_OK;
  });
}

llmCompletion_t *createCompletion(llmModel_t *model) {
  return runAndCatch<llmCompletion_t *>([model](){
    if (!model) throw lut::InvalidArgError("model");
    if (!model->model_for_generation) throw lut::InvalidArgError("model not initialized");

    std::unique_ptr<llmCompletion_t> comp = std::make_unique<llmCompletion_t>();
    comp->model_for_generation = model->model_for_generation;
    comp->tokenizer = model->tokenizer;
    comp->temperature = 1.0f;
    comp->top_k = 50;
    comp->top_p = 0.8f;
    
    return comp.release();
  }, nullptr);
}

llmStatus_t destroyCompletion(llmCompletion_t *comp) {
  delete comp;
  return LLM_OK;
}

llmStatus_t setPrompt(llmCompletion_t *comp, llmPrompt_t *prompt) {
  return runAndCatch([comp, prompt](){
    if (!comp) throw lut::InvalidArgError("comp");
    if (!prompt) throw lut::InvalidArgError("prompt");
    if (comp->generator) throw lut::InvalidArgError("completion already started");
    if (prompt->inputs.empty()) throw lut::InvalidArgError("prompt is empty");

    comp->prompt = prompt->inputs;
    return LLM_OK;
  });
}

llmStatus_t setTopP(llmCompletion_t *comp, float topP) {
  return runAndCatch([comp, topP](){
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->generator) throw lut::InvalidArgError("completion already started");

    comp->top_p = topP;
    return LLM_OK;
  });
}

llmStatus_t setTopK(llmCompletion_t *comp, int32_t topK) {
  return runAndCatch([comp, topK](){
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->generator) throw lut::InvalidArgError("completion already started");

    comp->top_k = topK;
    return LLM_OK;
  });
}

llmStatus_t setTemperature(llmCompletion_t *comp, float temperature) {
  return runAndCatch([comp, temperature](){
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->generator) throw lut::InvalidArgError("completion already started");

    comp->temperature = temperature;
    return LLM_OK;
  });
}

llmStatus_t startCompletion(llmCompletion_t *comp) {
  return runAndCatch([comp](){
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->generator) throw lut::InvalidArgError("completion already started");
    if (comp->prompt.empty()) throw lut::InvalidArgError("prompt is empty");

    std::shared_ptr<ModelForGeneration> model = comp->model_for_generation.lock();
    std::shared_ptr<Tokenizer> tokenizer = comp->tokenizer.lock();

    if (!model) throw lut::InvalidArgError("model had been destroyed");
    if (!tokenizer) throw lut::InvalidArgError("tokenizer had been destroyed");

    GenerationConfig config;
    config.temperature = comp->temperature;
    config.topK = comp->top_k;
    config.topP = comp->top_p;

    comp->generator = std::make_shared<Generator>(config, model, tokenizer);
    comp->generator->forwardPrompt(comp->prompt);
    return LLM_OK;
  });
}

llmBool_t isActive(llmCompletion_t *comp) {
  return runAndCatch<llmBool_t>([comp](){
    if (!comp) throw lut::InvalidArgError("comp");
    if (!comp->generator) throw lut::InvalidArgError("completion not started");

    return !comp->generator->stopped();
  }, false);
}

llmStatus_t getNextChunk(llmCompletion_t *comp, llmChunk_t *chunk) {
  return runAndCatch([comp, chunk](){
    if (!comp) throw lut::InvalidArgError("comp");
    if (!comp->generator) throw lut::InvalidArgError("completion not started");
    if (comp->generator->stopped()) throw lut::AbortedError("completion stopped");

    const char *token = comp->generator->nextToken();
    if (!token) throw lut::AbortedError("unexpected empty token");

    chunk->text = token;
    return LLM_OK;
  });
}

llmChunk_t *createChunk() {
  return new llmChunk_t();
}

llmStatus_t destroyChunk(llmChunk_t *chunk) {
  delete chunk;
  return LLM_OK;
}

const char *getChunkText(llmChunk_t *chunk) {
  return runAndCatch<const char *>([chunk](){
    if (!chunk) throw lut::InvalidArgError("chunk");
    return chunk->text.c_str();
  }, nullptr);
}


llmApi_t gApi{
  init,
  destroy,
  getLastErrorMessage,
  createModel,
  destroyModel,
  setModelConfig,
  setModelDevice,
  loadModel,
  getModelName,
  createPrompt,
  destroyPrompt,
  appendText,
  appendControlToken,
  createCompletion,
  destroyCompletion,
  setPrompt,
  setTopP,
  setTopK,
  setTemperature,
  startCompletion,
  isActive,
  getNextChunk,
  createChunk,
  destroyChunk,
  getChunkText
};

}  // namespace api
}  // namespace libllm

LLMAPI const llmApi_t *llmGetApi(int32_t version) {
  if (version == LLM_API_VERSION) return &libllm::api::gApi;
  return nullptr;
}

LLMAPI int32_t llmGetApiVersion() {
  return LLM_API_VERSION;
}
