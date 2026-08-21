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

#include "libllm/llama.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/constants.h"
#include "libllm/var_builder.h"
#include "lutil/ini_config.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "lutil/zip_file.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace libllm {
namespace llama {

namespace {

constexpr char TestCaseBin[] = "test_case.bin";
constexpr char ModelPackage[] = "llama3.2-3b-instruct-fp16.llmpkg";
constexpr char TestCasePackage[] = "llama3.2-3b-instruct-fp16_test.llmpkg";

// the reference is a fp32 huggingface forward while libllm runs in the default float type of the
// device, so the logits only have to be close.
constexpr double MaxRelDiff = 0.02;

// Prefill and single-token decode use different FP16 kernels; their drift grows with context.
constexpr double MaxLongContextRelDiff = 0.04;

// the packages are downloaded by tools/download_model_and_test_data.sh and are not part of the
// repository, so tests using them are skipped when they are missing.
std::string findPackage(const std::string &name) {
  for (const std::string &dir : {"models/", "../models/"}) {
    if (std::ifstream(dir + name, std::ios::binary).is_open()) return dir + name;
  }

  return "";
}

std::shared_ptr<LlamaModel> buildModel(
    const std::string &path,
  const fl::Device &device) {
  std::shared_ptr<lut::ZipFile> package = lut::ZipFile::fromFile(path);
  std::shared_ptr<lut::IniConfig> ini = lut::IniConfig::fromStream(
      package->open(ModelForGeneration::ModelConfig).get());

  std::string modelType = ini->getSection(ModelSection).getString(ModelTypeField);
  std::string modelFile = ini->getSection(ModelSection).getString(ModelFileField);
  LlamaConfig config = LlamaConfig::loadConfig(ini->getSection(modelType));

  VarBuilder vb = VarBuilder::fromStream(
      package->open(modelFile).get(),
      device,
      fl::F::getDefaultFloatType(device));

  return LlamaModel::build(config, vb.withName(modelType));
}

class TestLlamaModelForGeneration : public LlamaModelForGeneration {
 public:
  static std::shared_ptr<TestLlamaModelForGeneration> create(
      std::shared_ptr<LlamaModel> model,
      const fl::Device &device) {
    std::shared_ptr<TestLlamaModelForGeneration> result{new TestLlamaModelForGeneration()};
    result->_model = std::move(model);
    result->_config = result->_model->getConfig();
    result->_device = device;
    result->_floatType = fl::F::getDefaultFloatType(device);
    return result;
  }
};

std::shared_ptr<KVCacheManager> buildCacheWithBlocks(
    const LlamaConfig &config,
    const fl::Device &device,
    int numBlocks) {
  KVCacheSpec spec(
      config.numLayers,
      config.numKeyValueHeads,
      config.hiddenSize / config.numHeads,
      config.maxContextLength,
      fl::F::getDefaultFloatType(device));

  constexpr int BlockSize = 256;
  return std::make_shared<KVCacheManager>(spec, BlockSize, numBlocks, device);
}

// A block pool just large enough for one test sequence.
std::shared_ptr<KVCacheManager> buildCache(
    const LlamaConfig &config,
    const fl::Device &device,
    int numTokens) {
  constexpr int BlockSize = 256;
  int numBlocks = (numTokens + BlockSize - 1) / BlockSize;
  return buildCacheWithBlocks(config, device, numBlocks);
}

// `qLen` tokens appended after the `pastLen` tokens the cache already holds.
ForwardBatch makeBatch(
    const std::shared_ptr<KVCacheManager> &cache,
    const std::vector<int> &blockIds,
    int qLen,
    int pastLen,
    const fl::Device &device) {
  ForwardBatch batch = ForwardBatch::single(qLen, pastLen);
  batch.setKVCacheManager(cache);
  batch.setBlockIds({blockIds});
  batch.prepare(device);

  return batch;
}

ForwardBatch makePackedBatch(
    const std::shared_ptr<KVCacheManager> &cache,
    std::vector<fl::LongType> tokenIds,
    const std::vector<int> &queryLengths,
    const std::vector<int> &keyLengths,
    std::vector<fl::LongType> positionIds,
    std::vector<std::vector<int>> blockIds) {
  CATCH_REQUIRE(queryLengths.size() == keyLengths.size());

  std::vector<int> cuSeqlensQ(queryLengths.size() + 1, 0);
  std::vector<int> cuSeqlensK(keyLengths.size() + 1, 0);
  for (int i = 0; i < static_cast<int>(queryLengths.size()); ++i) {
    cuSeqlensQ[i + 1] = cuSeqlensQ[i] + queryLengths[i];
    cuSeqlensK[i + 1] = cuSeqlensK[i] + keyLengths[i];
  }

  ForwardBatch batch = ForwardBatch::packed(
      std::move(tokenIds),
      std::move(cuSeqlensQ),
      std::move(cuSeqlensK),
      std::move(positionIds));
  batch.setKVCacheManager(cache);
  batch.setBlockIds(std::move(blockIds));
  batch.prepare(cache->getKeyCache(0).getDevice());
  return batch;
}

VarBuilder loadTestCases(const std::string &path) {
  std::shared_ptr<lut::ZipFile> package = lut::ZipFile::fromFile(path);
  return VarBuilder::fromStream(
      package->open(TestCaseBin).get(),
      fl::Device::getCpu(),
      fl::DType::kFloat);
}

fl::Tensor toCpuFloat(fl::Tensor tensor) {
  tensor = fl::F::cast(tensor, fl::DType::kFloat);
  return fl::F::contiguous(fl::F::to(fl::Device::getCpu(), tensor));
}

// forward `inputIds` in one shot and return the fp32 logits of every position.
fl::Tensor forwardAll(
    const LlamaModel &model,
    fl::Tensor inputIds,
    const fl::Device &device) {
  fl::Tensor input = fl::F::to(device, inputIds);
  int numTokens = inputIds.getShape(0);

  std::shared_ptr<KVCacheManager> cache = buildCache(model.getConfig(), device, numTokens);
  std::vector<int> blockIds = cache->allocateBlocksForTokens(numTokens);

  return toCpuFloat(
      model.forwardLmHead(model.forward(input, makeBatch(cache, blockIds, numTokens, 0, device))));
}

int argmax(const float *data, int n) {
  return static_cast<int>(std::max_element(data, data + n) - data);
}

const float *rowOf(const fl::Tensor &logits, int row, int vocabSize) {
  const float *data = logits.getInternalData()->getData<float>(logits.getInternalOffset());
  return data + static_cast<int64_t>(row) * vocabSize;
}

// distance between `a` and the reference `b`, using the same definition as F::allClose.
double relDiff(const std::string &tag, const fl::Tensor &a, const fl::Tensor &b, int numEl) {
  const float *pa = a.getInternalData()->getData<float>(a.getInternalOffset());
  const float *pb = b.getInternalData()->getData<float>(b.getInternalOffset());

  double maxDiff = 0.0;
  double sumAbs = 0.0;
  for (int i = 0; i < numEl; ++i) {
    maxDiff = std::max(maxDiff, std::abs(static_cast<double>(pa[i]) - pb[i]));
    sumAbs += std::abs(static_cast<double>(pb[i]));
  }

  double meanAbs = sumAbs / numEl;
  double diff = maxDiff / meanAbs;
  LOG(INFO) << lut::sprintf(
      "%s: maxDiff=%.4f meanAbs=%.4f relDiff=%.5f",
      tag,
      maxDiff,
      meanAbs,
      diff);

  return diff;
}

std::vector<fl::Device> testDevices() {
  // the paged attention operators the model needs only exist for CUDA today.
  std::vector<fl::Device> devices;
  if (fl::isOperatorsAvailable(fl::Device::kCuda)) devices.emplace_back(fl::Device::getCuda());

  return devices;
}

}  // namespace

// the reference logits come from huggingface transformers, so a mismatch means libllm disagrees
// with transformers, not just with itself.
CATCH_TEST_CASE("test llama logits match the reference", "[libllm][llama]") {
  std::string modelPath = findPackage(ModelPackage);
  std::string testCasePath = findPackage(TestCasePackage);
  if (modelPath.empty() || testCasePath.empty()) {
    CATCH_SKIP("the model or the test case package not found in models/");
  }

  VarBuilder testCases = loadTestCases(testCasePath);

  // an empty package is a corrupted one, not a reason to silently check nothing.
  CATCH_REQUIRE(testCases.has("test_case.0.input_ids"));

  for (const fl::Device &device : testDevices()) {
    CATCH_INFO("device = " << device.getName());
    std::shared_ptr<LlamaModel> model = buildModel(modelPath, device);

    for (int caseIdx = 0;; ++caseIdx) {
      std::string prefix = lut::sprintf("test_case.%d.", caseIdx);
      if (!testCases.has(prefix + "input_ids")) break;

      CATCH_INFO("case = " << caseIdx);
      fl::Tensor inputIds = testCases.getUnchecked(prefix + "input_ids");
      fl::Tensor reference = testCases.getUnchecked(prefix + "logits");

      int numTokens = inputIds.getShape(0);
      int vocabSize = reference.getShape(1);
      CATCH_REQUIRE(reference.getShape(0) == numTokens);

      fl::Tensor logits = forwardAll(*model, inputIds, device);
      CATCH_REQUIRE(logits.getNumEl() == reference.getNumEl());

      CATCH_REQUIRE(
          relDiff(
              lut::sprintf("%s case=%d", device.getName(), caseIdx),
              logits,
              reference,
              numTokens * vocabSize) < MaxRelDiff);

      // the predicted token has to match exactly.
      for (int i = 0; i < numTokens; ++i) {
        CATCH_INFO("position = " << i);
        CATCH_REQUIRE(
            argmax(rowOf(logits, i, vocabSize), vocabSize) ==
            argmax(rowOf(reference, i, vocabSize), vocabSize));
      }
    }
  }
}

// Every incremental step has to match the corresponding row of a one-shot forward.
CATCH_TEST_CASE("test llama prefill matches incremental decode", "[libllm][llama]") {
  std::string modelPath = findPackage(ModelPackage);
  std::string testCasePath = findPackage(TestCasePackage);
  if (modelPath.empty() || testCasePath.empty()) {
    CATCH_SKIP("the model or the test case package not found in models/");
  }

  VarBuilder testCases = loadTestCases(testCasePath);
  fl::Tensor inputIds = testCases.getUnchecked("test_case.0.input_ids");
  int numTokens = inputIds.getShape(0);

  for (const fl::Device &device : testDevices()) {
    CATCH_INFO("device = " << device.getName());
    std::shared_ptr<LlamaModel> model = buildModel(modelPath, device);

    fl::Tensor prefillLogits = forwardAll(*model, inputIds, device);

    std::shared_ptr<KVCacheManager> cache = buildCache(model->getConfig(), device, numTokens);
    std::vector<int> blockIds = cache->allocateBlocksForTokens(numTokens);
    int vocabSize = prefillLogits.getShape(-1);
    for (int i = 0; i < numTokens; ++i) {
      CATCH_INFO("position = " << i);
      fl::Tensor token = fl::F::to(device, inputIds.slice({i, i + 1}));
      fl::Tensor hidden = model->forward(token, makeBatch(cache, blockIds, 1, i, device));
      fl::Tensor decodeLogits = toCpuFloat(model->forwardLmHead(hidden));

      fl::Tensor reference = prefillLogits.slice({i, i + 1});
      CATCH_REQUIRE(
          relDiff("incremental decode", decodeLogits, reference, vocabSize) < MaxRelDiff);
      CATCH_REQUIRE(
          argmax(rowOf(prefillLogits, i, vocabSize), vocabSize) ==
          argmax(rowOf(decodeLogits, 0, vocabSize), vocabSize));
    }
  }
}

CATCH_TEST_CASE("test llama packed batch matches independent requests", "[libllm][llama]") {
  std::string modelPath = findPackage(ModelPackage);
  std::string testCasePath = findPackage(TestCasePackage);
  if (modelPath.empty() || testCasePath.empty()) {
    CATCH_SKIP("the model or the test case package not found in models/");
  }

  VarBuilder testCases = loadTestCases(testCasePath);
  fl::Tensor seedIds = testCases.getUnchecked("test_case.0.input_ids");
  const fl::LongType *seed =
      seedIds.getInternalData()->getData<fl::LongType>(seedIds.getInternalOffset());

  std::vector<fl::LongType> promptA(seed, seed + 7);
  std::vector<fl::LongType> promptB(seed + 2, seed + 6);
  std::vector<fl::LongType> fullA = promptA;
  std::vector<fl::LongType> fullB = promptB;
  fullA.push_back(seed[7]);
  fullB.push_back(seed[6]);

  for (const fl::Device &device : testDevices()) {
    CATCH_INFO("device = " << device.getName());
    std::shared_ptr<LlamaModel> model = buildModel(modelPath, device);
    std::shared_ptr<TestLlamaModelForGeneration> generationModel =
        TestLlamaModelForGeneration::create(model, device);

    fl::Tensor promptATensor = fl::Tensor::create<fl::LongType>({7}, promptA);
    fl::Tensor promptBTensor = fl::Tensor::create<fl::LongType>({4}, promptB);
    fl::Tensor fullATensor = fl::Tensor::create<fl::LongType>({8}, fullA);
    fl::Tensor fullBTensor = fl::Tensor::create<fl::LongType>({5}, fullB);
    fl::Tensor promptAReference = forwardAll(*model, promptATensor, device);
    fl::Tensor promptBReference = forwardAll(*model, promptBTensor, device);
    fl::Tensor fullAReference = forwardAll(*model, fullATensor, device);
    fl::Tensor fullBReference = forwardAll(*model, fullBTensor, device);

    std::shared_ptr<KVCacheManager> cache = buildCacheWithBlocks(model->getConfig(), device, 2);
    std::vector<int> blocksA = cache->allocateBlocksForTokens(8);
    std::vector<int> blocksB = cache->allocateBlocksForTokens(5);

    std::vector<fl::LongType> prefillTokens = promptA;
    prefillTokens.insert(prefillTokens.end(), promptB.begin(), promptB.end());
    std::vector<fl::LongType> prefillPositions = {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3};
    ForwardBatch prefillBatch = makePackedBatch(
        cache,
        std::move(prefillTokens),
        {7, 4},
        {7, 4},
        std::move(prefillPositions),
        {blocksA, blocksB});
    fl::Tensor prefillLogits = toCpuFloat(generationModel->forward(prefillBatch));

    int vocabSize = model->getOutputDim();
    CATCH_REQUIRE(prefillLogits.getShape() == std::vector<int>{2, vocabSize});
    for (int sequence = 0; sequence < 2; ++sequence) {
      fl::Tensor reference = sequence == 0
          ? promptAReference.slice({6, 7})
          : promptBReference.slice({3, 4});
      CATCH_INFO("prefill sequence = " << sequence);
      CATCH_REQUIRE(
          relDiff(
              "packed prefill",
              prefillLogits.slice({sequence, sequence + 1}),
              reference,
              vocabSize) < MaxRelDiff);
      CATCH_REQUIRE(
          argmax(rowOf(prefillLogits, sequence, vocabSize), vocabSize) ==
          argmax(rowOf(reference, 0, vocabSize), vocabSize));
    }

    ForwardBatch decodeBatch = makePackedBatch(
        cache,
        {seed[7], seed[6]},
        {1, 1},
        {8, 5},
        {7, 4},
        {blocksA, blocksB});
    fl::Tensor decodeLogits = toCpuFloat(generationModel->forward(decodeBatch));

    CATCH_REQUIRE(decodeLogits.getShape() == std::vector<int>{2, vocabSize});
    for (int sequence = 0; sequence < 2; ++sequence) {
      fl::Tensor reference = sequence == 0
          ? fullAReference.slice({7, 8})
          : fullBReference.slice({4, 5});
      CATCH_INFO("decode sequence = " << sequence);
      CATCH_REQUIRE(
          relDiff(
              "packed decode",
              decodeLogits.slice({sequence, sequence + 1}),
              reference,
              vocabSize) < MaxRelDiff);
      CATCH_REQUIRE(
          argmax(rowOf(decodeLogits, sequence, vocabSize), vocabSize) ==
          argmax(rowOf(reference, 0, vocabSize), vocabSize));
    }
  }
}

CATCH_TEST_CASE("test llama decode crosses a KV cache block boundary", "[libllm][llama]") {
  std::string modelPath = findPackage(ModelPackage);
  std::string testCasePath = findPackage(TestCasePackage);
  if (modelPath.empty() || testCasePath.empty()) {
    CATCH_SKIP("the model or the test case package not found in models/");
  }

  VarBuilder testCases = loadTestCases(testCasePath);
  fl::Tensor seedIds = testCases.getUnchecked("test_case.0.input_ids");
  const fl::LongType *seed =
      seedIds.getInternalData()->getData<fl::LongType>(seedIds.getInternalOffset());
  int seedLength = seedIds.getShape(0);

  constexpr int NumTokens = 257;
  constexpr int FirstComparedPosition = 254;
  std::vector<fl::LongType> tokenIds(NumTokens);
  for (int i = 0; i < NumTokens; ++i) tokenIds[i] = seed[i % seedLength];
  fl::Tensor inputIds = fl::Tensor::create<fl::LongType>({NumTokens}, tokenIds);

  for (const fl::Device &device : testDevices()) {
    CATCH_INFO("device = " << device.getName());
    std::shared_ptr<LlamaModel> model = buildModel(modelPath, device);

    std::shared_ptr<KVCacheManager> prefillCache =
        buildCache(model->getConfig(), device, NumTokens);
    std::vector<int> prefillBlocks = prefillCache->allocateBlocksForTokens(NumTokens);
    fl::Tensor input = fl::F::to(device, inputIds);
    fl::Tensor prefillHidden = toCpuFloat(
      model->forward(input, makeBatch(prefillCache, prefillBlocks, NumTokens, 0, device)));

    std::shared_ptr<KVCacheManager> decodeCache =
        buildCache(model->getConfig(), device, NumTokens);
    std::vector<int> decodeBlocks = decodeCache->allocateBlocksForTokens(NumTokens);
    for (int i = 0; i < NumTokens; ++i) {
      fl::Tensor token = input.slice({i, i + 1});
      fl::Tensor decodeHidden =
          model->forward(token, makeBatch(decodeCache, decodeBlocks, 1, i, device));
      if (i < FirstComparedPosition) continue;

      CATCH_INFO("position = " << i);
      fl::Tensor reference = prefillHidden.slice({i, i + 1});
          CATCH_REQUIRE(
          relDiff(
              "KV block boundary",
              toCpuFloat(decodeHidden),
              reference,
              model->getConfig().hiddenSize) < MaxLongContextRelDiff);
    }
  }
}

}  // namespace llama
}  // namespace libllm
