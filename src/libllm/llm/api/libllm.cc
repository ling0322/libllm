#include "llm/api/llm.h"

#include <memory>
#include "llyn/llyn.h"
#include "llm/api/environment.h"
#include "llm/api/model_factory.h"
#include "llm/chatglm2/chatglm2_model.h"
#include "llm/common/generator.h"
#include "llm/common/model_for_generation.h"
#include "lytok/lytok.h"
#include "lyutil/error.h"
#include "lyutil/ini_config.h"
#include "lyutil/log.h"

using libllm::Environment;
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
  std::shared_ptr<ModelForGeneration> model_for_generation;
  std::shared_ptr<Tokenizer> tokenizer;

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

LLM_RESULT llm_init() {
  try {
    Environment::init();
    return LLM_OK;
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return static_cast<LLM_RESULT>(e.getCode());
  }
}

void llm_destroy() {
  Environment::destroy();
}

llm_model_t *llm_model_init(const char *ini_path) {
  std::unique_ptr<llm_model_t> model = std::make_unique<llm_model_t>();
  try {
    if (!ini_path)
      throw ly::InvalidArgError("ini_path");
    std::unique_ptr<IniConfig> ini = IniConfig::read(ini_path);

    model->ctx.setDevice(llyn::Device::createForCPU());
    model->tokenizer = Tokenizer::create(ini->getSection("tokenizer"));
    model->model_for_generation = ModelFactory::createModel(model->ctx, *ini);
  
    return model.release();
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return nullptr;
  }
}

const char *llm_model_get_name(llm_model_t *m) {
  try {
    if (!m) throw ly::InvalidArgError("m");
    if (!m->model_for_generation) throw ly::InvalidArgError("m");

    return m->model_for_generation->getName();
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return nullptr;
  }
}

void llm_model_destroy(llm_model_t *m) {
  delete m;
}

llm_compl_opt_t::llm_compl_opt_t() :
    top_k(50),
    top_p(0.8f),
    temperature(1.0f) {}

llm_compl_opt_t *llm_compl_opt_init() {
  try {
    return new llm_compl_opt_t();
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return nullptr;
  }
}
void llm_compl_opt_destroy(llm_compl_opt_t *o) {
  delete o;
}

LLM_RESULT llm_compl_opt_set_top_p(llm_compl_opt_t *o, float topp) {
  try {
    if (!o) throw ly::InvalidArgError("o");
    if (topp > 1.0f) throw ly::InvalidArgError("topp");
    o->top_p = topp;
    return LLM_OK;
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return static_cast<LLM_RESULT>(e.getCode());
  }
}
LLM_RESULT llm_compl_opt_set_temperature(
    llm_compl_opt_t *o, float temperature) {
  try {
    if (!o) throw ly::InvalidArgError("o");
    if (temperature <= 0.0f) throw ly::InvalidArgError("temperature");
    o->temperature = temperature;
    return LLM_OK;
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return static_cast<LLM_RESULT>(e.getCode());
  }
}
LLM_RESULT llm_compl_opt_set_prompt(llm_compl_opt_t *o, const char *prompt) {
  try {
    if (!o) throw ly::InvalidArgError("o");
    if (!prompt) throw ly::InvalidArgError("prompt");
    o->prompt = prompt;
    return LLM_OK;
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return static_cast<LLM_RESULT>(e.getCode());
  }
}
LLM_RESULT llm_compl_opt_set_top_k(llm_compl_opt_t *o, int32_t topk) {
  try {
    if (!o) throw ly::InvalidArgError("o");
    if (topk <= 0) throw ly::InvalidArgError("topk");
    o->top_k = topk;
    return LLM_OK;
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return static_cast<LLM_RESULT>(e.getCode());
  }
}

LLM_RESULT llm_compl_opt_set_model(
    llm_compl_opt_t *o, llm_model_t *model) {
  try {
    if (!o) throw ly::InvalidArgError("o");
    if (!model) throw ly::InvalidArgError("model");
    o->model_for_generation = model->model_for_generation;
    o->tokenizer = model->tokenizer;
    return LLM_OK;
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return static_cast<LLM_RESULT>(e.getCode());
  }
}

llm_compl_t *llm_compl_init(llm_compl_opt_t *o) {
  std::unique_ptr<llm_compl_t> c = std::make_unique<llm_compl_t>();
  try {
    if (!o) throw ly::InvalidArgError("o");
    if (o->prompt == "") throw ly::InvalidArgError("prompt");
    if ((!o->model_for_generation) || (!o->tokenizer)) throw ly::InvalidArgError("model");

    GenerationConfig config;
    config.temperature = o->temperature;
    config.topK = o->top_k;
    config.topP = o->top_p;

    c->generator = std::make_shared<Generator>(config, o->model_for_generation, o->tokenizer);
    c->generator->setPrompt(o->prompt);
    
    return c.release();
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return nullptr;
  }
}

LLM_BOOL llm_compl_stopped(llm_compl_t *c) {
  return c->generator->stopped() ? LLM_TRUE : LLM_FALSE;
}

void llm_compl_destroy(llm_compl_t *c) {
  delete c;
}

llm_chunk_t *llm_compl_next_chunk(llm_compl_t *c) {
  std::unique_ptr<llm_chunk_t> chunk = std::make_unique<llm_chunk_t>();
  try {
    if (!c) throw ly::InvalidArgError("c");
    const char *token = c->generator->nextToken();
    if (!token) {
      return nullptr;
    }

    chunk->text = token;
    return chunk.release();
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return nullptr;
  }
}

const char *llm_chunk_get_text(llm_chunk_t *c) {
  try {
    if (!c) throw ly::InvalidArgError("c");
    return c->text.c_str();
  } catch (const ly::Error &e) {
    LOG(ERROR) << e.what();
    return nullptr;
  }
}

void llm_chunk_destroy(llm_chunk_t *c) {
  delete c;
}
