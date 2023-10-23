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

#include "catch2/catch_amalgamated.hpp"
#include "llyn/util.h"
#include "lyutil/ini_config.h"
#include "lyutil/strings.h"
#include "lyutil/time.h"
#include "lytok/lytok.h"
#include "llm/common/generator.h"
#include "llm/chatglm2/chatglm2_config.h"
#include "llm/chatglm2/chatglm2_model.h"


using llyn::Context;
using llyn::LongType;
using llyn::StateMap;
using llyn::Tensor;
using llyn::getCtxForCPU;
using llyn::readParameters;
using libllm::GenerationConfig;
using libllm::Generator;
using libllm::chatglm2::ChatGLM2Config;
using libllm::chatglm2::ChatGLM2Model;
using lytok::Tokenizer;

namespace F = llyn::functional;

std::shared_ptr<ChatGLM2Model> readModel(const std::string &iniPath) {
  Context ctx = getCtxForCPU();
  auto ini = ly::IniConfig::read(iniPath);
  auto config = ChatGLM2Config::fromIni(*ini);
  auto model = ChatGLM2Model::create(ctx, *config);

  ly::Path modelPath = ini->getSection("model").getPath("model_file");
  readParameters(modelPath.string(), model.get());

  return model;
}

CATCH_TEST_CASE("test ChatGLM2 module", "[core][nn][chatglm2]") {
  ly::Path modelDir = ly::Path("data") / "test";
  ly::Path iniPath = modelDir / "chatglm2.config.ini";
  auto model = readModel(iniPath.string());

  Tensor in = Tensor::create<LongType>({1, 18}, {
      64790, 64792, 790, 30951, 517, 30910, 30939, 30996, 13, 13, 54761, 31211, 39701, 13, 13,
      55437, 31211, 36474
  });

  StateMap stateMap;
  Tensor x = model->forward(&stateMap, in);
  F::print(x);
}

void benchmarkDecoding(const ChatGLM2Model *model, int ctxLen) {
  std::vector<LongType> ctx = {
      64790, 64792, 790, 30951, 517, 30910, 30939, 30996, 13, 13, 54761, 31211, 39701, 13, 13,
      55437, 31211 };
  while (ctx.size() < ctxLen) {
    ctx.push_back(39701);
  }

  Tensor in = Tensor::create<LongType>({1, ctxLen}, ctx);

  StateMap stateMap;
  double t0 = ly::now();
  Tensor x = model->forward(&stateMap, in);
  double t1 = ly::now() - t0;


  in = Tensor::create<LongType>({1, 1}, {39701});


  t0 = ly::now();
  x = model->forward(&stateMap, in);
  double t2 = ly::now() - t0;
  LOG(INFO) << ly::sprintf(
      "ctxLen=%d forward=%.2fms decode=%.2fms", ctxLen, 1000.0 * t1, 1000.0 * t2);
}

CATCH_TEST_CASE("benchmark ChatGLM2 q4sym decoding", "[llmpp][chatglm2][benchmark]") {
  ly::Path modelDir = ly::Path("data") / "test";
  ly::Path iniPath = modelDir / "chatglm2.q4sym.ini";
  auto model = readModel(iniPath.string());

  benchmarkDecoding(model.get(), 50);
  benchmarkDecoding(model.get(), 200);
  benchmarkDecoding(model.get(), 500);
}

CATCH_TEST_CASE("test ChatGLM2 generation", "[llmpp][chatglm2][generation]") {
  ly::Path modelDir = ly::Path("data") / "test";
  ly::Path iniFile = modelDir / "chatglm2.tokenizer.ini";

  std::unique_ptr<ly::IniConfig> ini = ly::IniConfig::read(iniFile.string());
  std::shared_ptr<Tokenizer> tokenizer = Tokenizer::create(ini->getSection("tokenizer"));

  iniFile = modelDir / "chatglm2.config.ini";
  auto model = readModel(iniFile.string());

  GenerationConfig generationConfig;
  Generator generator(generationConfig, model, tokenizer);
  generator.setPrompt("[Round 1]\n\n问：你好\n\n答：");

  fputc('\n', stdout);
  const char *tok = nullptr;
  while ((tok = generator.nextToken()) != nullptr) {
    fputs(tok, stdout);
    fflush(stdout);
  } 
  fputc('\n', stdout);
}
