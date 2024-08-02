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
#include "libllm/model_for_generation.h"
#include "libllm/operators.h"
#include "lut/error.h"
#include "lut/flags.h"
#include "lut/random.h"
#include "lut/time.h"

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
float benchmarkPromptForward(
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
    StateMap past;
    x = model->forward(past, inputs);
    ++nLoop;
  }
  double t1 = lut::now();
  int numToken = promptLen * nLoop;
  float tokenPerSec = numToken / (t1 - t0);

  return tokenPerSec;
}

float benchmarkTokenGeneration(
    lut::Random *r,
    std::shared_ptr<llama::LlamaModel> model,
    int vocabSize,
    int contextLen) {
  // get kv_cache for the given context size.
  StateMap past;
  Tensor inputs = randomLongTensor(r, contextLen, vocabSize, model->getCtx().getDevice());
  Tensor x = model->forward(past, inputs);

  // first run.
  StateMap pastClone = past.clone();
  std::array<LongType, 1> inputData{1024};
  Tensor inputToken = Tensor::create<LongType>({1, 1}, inputData);
  inputToken = F::to(model->getCtx().getDevice(), inputToken);

  x = model->forward(pastClone, inputToken);
  x = model->forwardLmHead(x);

  double t0 = lut::now();
  int nLoop = 0;
  while (lut::now() - t0 < MaxWait) {
    StateMap pastClone = past.clone();
    x = model->forward(pastClone, inputToken);
    x = model->forwardLmHead(x);
    ++nLoop;
  }
  double t1 = lut::now();
  int numToken = nLoop;
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
    config.numKeyValueHeads = 32;
    config.numLayers = 32;
    config.qkvProjBias = false;
    config.vocabSize = 32000;

    return config;
  }

  NOT_IMPL();
}

std::shared_ptr<llama::LlamaModel> getLlamaModel(
    lut::Random *r,
    LlamaType type,
    Device device,
    DType weightType) {
  Context ctx;
  ctx.setDevice(device);
  ctx.setFloatDType(F::getDefaultFloatType(device));

  llama::LlamaConfig config = getLlamaConfig(type);
  std::shared_ptr<llama::LlamaModel> model = llama::LlamaModel::create(ctx, config);
  model->initParameters(r, weightType);

  return model;
}

void benchmarkLlama(std::shared_ptr<llama::LlamaModel> model, int ctxLength, DType weightType) {
  lut::Random r(MagicNumber);

  float tokenPerSec = benchmarkPromptForward(&r, model, 32000, ctxLength);
  printf(
      "llama2_7B   %-8s %-8s prompt@len:%-5d   %-7.1f\n",
      model->getCtx().getDevice().getName().c_str(),
      weightType.toString().c_str(),
      ctxLength,
      tokenPerSec);

  tokenPerSec = benchmarkTokenGeneration(&r, model, 32000, ctxLength);
  printf(
      "llama2_7B   %-8s %-8s tokengen@ctx:%-5d %-7.1f\n",
      model->getCtx().getDevice().getName().c_str(),
      weightType.toString().c_str(),
      ctxLength,
      tokenPerSec);
}

int benchmarkMain(Device device) {
  libllm::initOperators();

  LlamaType llamaType = LlamaType::Llama2_7B;
  DType weightType = libllm::DType::kQInt4x32;

  LOG(INFO) << "intializing model ...";
  auto model = libllm::getLlamaModel(nullptr, llamaType, device, weightType);
  LOG(INFO) << "model initialized.";

  printf("==========================================================\n");
  printf("ModelType   Device   Weight   Task               Token/s  \n");
  printf("----------------------------------------------------------\n");

  libllm::benchmarkLlama(model, 128, libllm::DType::kQInt4x32);
  libllm::benchmarkLlama(model, 512, libllm::DType::kQInt4x32);

  printf("----------------------------------------------------------\n");

  libllm::destroyOperators();
  return 0;
}

}  // namespace libllm

int main(int argc, char **argv) {
  const char *usage =
      "Command line interface for benchmarking libllm.\n"
      "Usage: benchmark [-d (cpu|cuda)]";

  std::string deviceType = "cuda";
  lut::Flags flags(usage);
  flags.define("-d", &deviceType, "device of the model. (cpu|cuda)");
  flags.parse(argc, argv);

  if (deviceType == "cpu") {
    libllm::benchmarkMain(libllm::Device::getCpu());
    return 0;
  } else if (deviceType == "cuda") {
    libllm::benchmarkMain(libllm::Device::getCuda());
    return 0;
  } else {
    fprintf(stderr, "unexpected device %s\n", deviceType.c_str());
    return 1;
  }
}
