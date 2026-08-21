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

#include <memory>
#include <string>

#include "libllm/kv_cache.h"
#include "libllm/model_for_generation.h"
#include "libllm/forward_batch.h"
#include "libllm/prompt.h"
#include "lutil/error.h"
#include "lutil/flags.h"
#include "lutil/time.h"
#include "lutil/zip_file.h"
#include "flint/functional.h"
#include "flint/operators.h"

constexpr double MaxWait = 10;

namespace libllm {

Prompt makePrompt(int repeatCount) {
  CHECK(repeatCount > 0);

  std::string text;
  for (int i = 0; i < repeatCount; ++i) {
    text += "The quick brown fox jumps over the lazy dog. ";
  }

  Prompt prompt;
  prompt.appendText(text);
  return prompt;
}

// `tokenIds` appended after the `pastLen` tokens `blockIds` already holds. Re-running the same
// batch overwrites the same slots, which is what lets the loops below repeat without resetting.
ForwardBatch makeBatch(
    const std::shared_ptr<ModelForGeneration> &model,
    const std::vector<int> &blockIds,
    lut::Span<const fl::LongType> tokenIds,
    int pastLen) {
  ForwardBatch batch = ForwardBatch::single(tokenIds, pastLen);
  batch.setKVCacheManager(model->getKVCacheManager());
  batch.setBlockIds({blockIds});
  batch.prepare(model->getDevice());

  return batch;
}

float benchmarkPrefill(
    const std::shared_ptr<ModelForGeneration> &model,
    const Prompt &prompt,
    int promptTokenCount) {
  std::vector<fl::LongType> tokenIds = model->encodePrompt(prompt);
  std::shared_ptr<KVCacheManager> manager = model->getKVCacheManager().lock();
  CHECK(manager);

  std::vector<int> blockIds = manager->allocateBlocksForTokens(
      static_cast<int>(tokenIds.size()));
  CHECK(!blockIds.empty());

  model->forward(makeBatch(model, blockIds, lut::makeConstSpan(tokenIds), 0));

  double start = lut::now();
  int loops = 0;
  while (lut::now() - start < MaxWait) {
    model->forward(makeBatch(model, blockIds, lut::makeConstSpan(tokenIds), 0));
    ++loops;
  }
  double elapsed = lut::now() - start;

  manager->freeBlocks(blockIds);
  return promptTokenCount * loops / elapsed;
}

float benchmarkDecode(const std::shared_ptr<ModelForGeneration> &model, const Prompt &prompt) {
  std::vector<fl::LongType> tokenIds = model->encodePrompt(prompt);
  int promptLength = static_cast<int>(tokenIds.size());
  std::shared_ptr<KVCacheManager> manager = model->getKVCacheManager().lock();
  CHECK(manager);

  std::vector<int> blockIds = manager->allocateBlocksForTokens(promptLength + 1);
  CHECK(!blockIds.empty());

  model->forward(makeBatch(model, blockIds, lut::makeConstSpan(tokenIds), 0));

  std::array<fl::LongType, 1> decodeToken{0};
  model->forward(makeBatch(model, blockIds, decodeToken, promptLength));

  double start = lut::now();
  int loops = 0;
  while (lut::now() - start < MaxWait) {
    model->forward(makeBatch(model, blockIds, decodeToken, promptLength));
    ++loops;
  }
  double elapsed = lut::now() - start;

  manager->freeBlocks(blockIds);
  return loops / elapsed;
}

int benchmarkMain(fl::Device device, const std::string &modelPath, int promptRepeatCount) {
  fl::initOperators();

  std::shared_ptr<lut::ZipFile> package = lut::ZipFile::fromFile(modelPath);
  std::shared_ptr<ModelForGeneration> model = ModelForGeneration::fromPackage(
      device,
      package.get());
  Prompt prompt = makePrompt(promptRepeatCount);
  int promptTokenCount = model->getPromptTokenCount(prompt);

  float prefillTokensPerSecond = benchmarkPrefill(model, prompt, promptTokenCount);
  printf(
      "%-12s %-8s prefill@len:%-5d  %-7.1f tokens/s\n",
      model->getName(),
      device.getName().c_str(),
      promptTokenCount,
      prefillTokensPerSecond);

  float decodeTokensPerSecond = benchmarkDecode(model, prompt);
  printf(
      "%-12s %-8s decode@ctx:%-5d   %-7.1f tokens/s\n",
      model->getName(),
      device.getName().c_str(),
      promptTokenCount,
      decodeTokensPerSecond);

  model.reset();
  package.reset();
  fl::destroyOperators();
  return 0;
}

}  // namespace libllm

int main(int argc, char **argv) {
  const char *usage =
      "Benchmark model prefill and decode throughput.\n"
      "Usage: llm_benchmark -m <model.llmpkg> [-d (cpu|cuda)] [-l prompt-repeat-count]";

  std::string deviceType = "cuda";
  std::string modelPath;
  std::string promptRepeatCountString = "16";
  lut::Flags flags(usage);
  flags.define("-d", &deviceType, "device of the model. (cpu|cuda)");
  flags.define("-m", &modelPath, "path to the model package.");
  flags.define("-l", &promptRepeatCountString, "number of prompt sentence repetitions.");
  flags.parse(argc, argv);

  if (modelPath.empty()) {
    flags.printUsage();
    return 1;
  }

  int promptRepeatCount = std::stoi(promptRepeatCountString);
  if (deviceType == "cpu") {
    return libllm::benchmarkMain(fl::Device::getCpu(), modelPath, promptRepeatCount);
  } else if (deviceType == "cuda") {
    return libllm::benchmarkMain(fl::Device::getCuda(), modelPath, promptRepeatCount);
  }

  fprintf(stderr, "unexpected device %s\n", deviceType.c_str());
  return 1;
}