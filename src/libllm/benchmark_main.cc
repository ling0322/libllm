// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include <stdio.h>

#include <iostream>
#include <string>

#include "libllm/dtype.h"
#include "libllm/functional.h"
#include "libllm/llama.h"
#include "libllm/llm.h"
#include "libllm/lut/error.h"
#include "libllm/lut/flags.h"
#include "libllm/lut/random.h"
#include "libllm/lut/time.h"
#include "libllm/model_for_generation.h"

constexpr int MagicNumber = 0x55aa;
constexpr double MaxWait = 10;

enum class LlamaType { Llama2_7B };

namespace libllm {

Tensor randomLongTensor(lut::Random *r, int length, int vocabSize, Device device) {
  std::vector<LongType> prompt(length);
  r->fill(lut::makeSpan(prompt), 0, vocabSize);

  Tensor inputs = Tensor::create<LongType>({1, length}, prompt);
  inputs = F::to(device, inputs);

  return inputs;
}

/// @brief Benchmark the prompt forwarding time.
/// @param model The model to benchmark.
/// @param vocabSize Vocabulary size of the model.
/// @param promptLen length of the prompt to evaluate.
/// @return number of tokens per second.
float bemchmarkPromptForward(
    lut::Random *r,
    std::shared_ptr<llama::LlamaModel> model,
    int vocabSize,
    int promptLen) {
  // first run.
  StateMap past;
  Tensor inputs = randomLongTensor(r, promptLen, vocabSize, model->getCtx().getDevice());
  Tensor x = model->forward(past, inputs);

  double t0 = lut::now();
  int nLoop = 0;
  while (lut::now() - t0 < MaxWait) {
    LOG(INFO) << lut::now() - t0;
    StateMap past;
    x = model->forward(past, inputs);
    ++nLoop;
  }
  double t1 = lut::now();
  int numToken = promptLen * nLoop;
  float tokenPerSec = numToken / (t1 - t0);

  return tokenPerSec;
}

llama::LlamaConfig getLlamaConfig(LlamaType type) {
  llama::LlamaConfig config;
  if (type == LlamaType::Llama2_7B) {
    config.hiddenSize = 4096;
    config.intermediateSize = 11008;
    config.maxContextLength = 4096;
    config.normEps = 1e-5;
    config.numHeads = 32;
    config.numLayers = 32;
    config.qkvProjBias = false;
    config.vocabSize = 32000;

    return config;
  }

  NOT_IMPL();
}

std::shared_ptr<llama::LlamaModel>
getLlamaModel(lut::Random *r, LlamaType type, Device device, DType weightType) {
  Context ctx;
  ctx.setDevice(device);
  ctx.setFloatDType(F::getDefaultFloatType(device));

  llama::LlamaConfig config = getLlamaConfig(type);
  std::shared_ptr<llama::LlamaModel> model = llama::LlamaModel::create(ctx, config);
  model->initParameters(r, weightType);

  return model;
}

void benchmarkLlama(LlamaType llamaType, int ctxLength, Device device, DType weightType) {
  lut::Random r(MagicNumber);

  LOG(INFO) << "intializing model ...";
  std::shared_ptr<llama::LlamaModel> model = getLlamaModel(nullptr, llamaType, device, weightType);
  LOG(INFO) << "model initialized.";
  LOG(INFO) << "start benchmarking ...";
  float tokenPerSec = bemchmarkPromptForward(&r, model, 32000, ctxLength);

  printf(
      "llama2_7B       %10s%10sprompt@%d   %3.2f\n",
      device.getName().c_str(),
      weightType.toString().c_str(),
      ctxLength,
      tokenPerSec);
}

}  // namespace libllm

int main(int argc, char **argv) {
  CHECK(llmInit(LLM_API_VERSION) == LLM_OK);

  libllm::benchmarkLlama(
      LlamaType::Llama2_7B,
      128,
      libllm::Device::getCuda(),
      libllm::DType::kQInt4x32);

  return 0;
}
