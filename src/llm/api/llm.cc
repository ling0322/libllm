#include "llm/api/llm.h"

#include <string.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include "llyn/llyn.h"
#include "llm/api/model_factory.h"
#include "llm/chatglm2/chatglm2_model.h"
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
using libllm::chatglm2::ChatGLM2Config;
using libllm::chatglm2::ChatGLM2Model;
using lytok::Tokenizer;
using ly::IniConfig;

struct llm_model_t {
  llyn::Context ctx;
  std::shared_ptr<ModelForGeneration> model_for_generation;
  std::shared_ptr<Tokenizer> tokenizer;
};

struct llm_compl_opt_t {
  int top_k;
  float top_p;
  float temperature;
  std::string prompt;

  llm_compl_opt_t();
};

struct llm_compl_t {
  std::shared_ptr<Generator> generator;
};

struct llm_chunk_t {
  std::string text;
};

namespace {

thread_local int gErrorCode = static_cast<int>(ly::ErrorCode::OK);
thread_local char gErrorMessage[512] = "";
static std::atomic<bool> gInitialized{false};

void setErrorCodeAndMessage(const ly::Error &e) {
  gErrorCode = static_cast<int>(e.getCode());

  std::string what = e.what();
  if (what.size() >= sizeof(gErrorMessage)) {
    what.erase(what.begin() + sizeof(gErrorMessage) - 4, what.end());
    what += "...";
  }
  snprintf(gErrorMessage, sizeof(gErrorMessage), "%s", what.c_str());
}

LL_STATUS runAndCatch(std::function<void()> &&f) {
  try {
    f();
    return LL_OK;
  } catch (const ly::Error &e) {
    setErrorCodeAndMessage(e);
    return static_cast<LL_STATUS>(e.getCode());
  }
}

template<typename T>
T runAndCatch(std::function<T()> &&c, T default_value) {
  try {
    return c();
  } catch (const ly::Error &e) {
    setErrorCodeAndMessage(e);
    return default_value;
  }
}
 
}  // anonymous namespace

LL_STATUS llm_init() {
  if (!gInitialized.exchange(true)) {
    try {
      ly::setLogLevel(ly::LogSeverity::INFO);
      llyn::init();

      return LL_OK;
    } catch (const ly::Error &e) {
      gInitialized = false;
      setErrorCodeAndMessage(e);
      return static_cast<LL_STATUS>(e.getCode());;
    }
  }

  return LL_OK;
}

void llm_destroy() {
  if (gInitialized.exchange(false)) {
    llyn::destroy();
  }
}

llm_model_t *llm_model_init(const char *ini_path) {
  return runAndCatch<llm_model_t *>([ini_path](){
    std::unique_ptr<llm_model_t> model = std::make_unique<llm_model_t>();

    if (!ini_path)
      throw ly::InvalidArgError("ini_path");
    std::unique_ptr<IniConfig> ini = IniConfig::read(ini_path);

    model->ctx.setDevice(llyn::Device::createForCPU());
    model->tokenizer = Tokenizer::create(ini->getSection("tokenizer"));
    model->model_for_generation = ModelFactory::createModel(model->ctx, *ini);
  
    return model.release();
  }, nullptr);
}

const char *llm_model_get_name(llm_model_t *m) {
  return runAndCatch<const char *>([m](){
    if (!m) throw ly::InvalidArgError("m");
    if (!m->model_for_generation) throw ly::InvalidArgError("m");

    return m->model_for_generation->getName();
  }, nullptr);
}

llm_compl_t *llm_model_complete(llm_model_t *m, llm_compl_opt_t *o) {
  return runAndCatch<llm_compl_t *>([m, o](){
    if (!o) throw ly::InvalidArgError("o");
    if (!m) throw ly::InvalidArgError("m");

    if (o->prompt == "") throw ly::InvalidArgError("prompt");
    if ((!m->model_for_generation) || (!m->tokenizer)) throw ly::InvalidArgError("model");

    GenerationConfig config;
    config.temperature = o->temperature;
    config.topK = o->top_k;
    config.topP = o->top_p;

    std::unique_ptr<llm_compl_t> c = std::make_unique<llm_compl_t>();
    c->generator = std::make_shared<Generator>(config, m->model_for_generation, m->tokenizer);
    c->generator->setPrompt(o->prompt);
    
    return c.release();
  }, nullptr);
}

void llm_model_destroy(llm_model_t *m) {
  delete m;
}

llm_compl_opt_t::llm_compl_opt_t() :
    top_k(50),
    top_p(0.8f),
    temperature(1.0f) {}

llm_compl_opt_t *llm_compl_opt_init() {
  return new llm_compl_opt_t();
}

void llm_compl_opt_destroy(llm_compl_opt_t *o) {
  delete o;
}

LL_STATUS llm_compl_opt_set_top_p(llm_compl_opt_t *o, float topp) {
  return runAndCatch([o, topp](){
    if (!o) throw ly::InvalidArgError("o");
    if (topp > 1.0f) throw ly::InvalidArgError("topp");
    o->top_p = topp;
  });
}

LL_STATUS llm_compl_opt_set_temperature(llm_compl_opt_t *o, float temperature) {
  return runAndCatch([o, temperature](){
    if (!o) throw ly::InvalidArgError("o");
    if (temperature <= 0.0f) throw ly::InvalidArgError("temperature");
    o->temperature = temperature;
  });
}

LL_STATUS llm_compl_opt_set_prompt(llm_compl_opt_t *o, const char *prompt) {
  return runAndCatch([o, prompt](){
    if (!o) throw ly::InvalidArgError("o");
    if (!prompt) throw ly::InvalidArgError("prompt");
    o->prompt = prompt;
  });
}

LL_STATUS llm_compl_opt_set_top_k(llm_compl_opt_t *o, int32_t topk) {
  return runAndCatch([o, topk](){
    if (!o) throw ly::InvalidArgError("o");
    if (topk <= 0) throw ly::InvalidArgError("topk");
    o->top_k = topk;
  });
}

LL_BOOL llm_compl_is_active(llm_compl_t *c) {
  return runAndCatch<LL_BOOL>([c](){
    if (!c) throw ly::InvalidArgError("c");
    return c->generator->stopped() ? LL_FALSE : LL_TRUE;
  }, LL_FALSE);
}

void llm_compl_destroy(llm_compl_t *c) {
  delete c;
}

llm_chunk_t *llm_compl_next_chunk(llm_compl_t *c) {  
  return runAndCatch<llm_chunk_t *>([c](){
    if (!c) throw ly::InvalidArgError("c");
    if (c->generator->stopped()) throw ly::AbortedError("call next() on stopped completion.");

    const char *token = c->generator->nextToken();
    if (!token) throw ly::AbortedError("unexpected empty token");

    std::unique_ptr<llm_chunk_t> chunk = std::make_unique<llm_chunk_t>();
    chunk->text = token;
    return chunk.release();
  }, nullptr);
}

const char *llm_chunk_get_text(llm_chunk_t *c) {
  return runAndCatch<const char *>([c](){
    if (!c) throw ly::InvalidArgError("c");
    return c->text.c_str();
  }, nullptr);
}

void llm_chunk_destroy(llm_chunk_t *c) {
  delete c;
}

const char *llm_get_last_error_message() {
  return gErrorMessage;
}
