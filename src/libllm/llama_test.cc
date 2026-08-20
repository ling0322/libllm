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

// the packages are downloaded by tools/download_model_and_test_data.sh and are not part of the
// repository, so tests using them are skipped when they are missing.
std::string findPackage(const std::string &name) {
  for (const std::string &dir : {"models/", "../models/"}) {
    if (std::ifstream(dir + name, std::ios::binary).is_open()) return dir + name;
  }

  return "";
}

std::shared_ptr<LlamaModel> buildModel(const std::string &path, const fl::Device &device) {
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
fl::Tensor forwardAll(const LlamaModel &model, fl::Tensor inputIds, const fl::Device &device) {
  fl::Tensor input = fl::F::to(device, inputIds);

  KVCache past;
  return toCpuFloat(model.forwardLmHead(model.forward(past, input)));
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
  std::vector<fl::Device> devices{fl::Device::getCpu()};
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

// forwarding the whole input at once has to predict the same next token as forwarding it one token
// at a time. It covers the kv cache append, the rope offset and the causal mask alignment.
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

    KVCache past;
    fl::Tensor hidden;
    for (int i = 0; i < numTokens; ++i) {
      fl::Tensor token = fl::F::to(device, inputIds.slice({i, i + 1}));
      hidden = model->forward(past, token);
    }
    fl::Tensor decodeLogits = toCpuFloat(model->forwardLmHead(hidden));

    int vocabSize = decodeLogits.getShape(-1);
    CATCH_REQUIRE(
        argmax(rowOf(prefillLogits, numTokens - 1, vocabSize), vocabSize) ==
        argmax(rowOf(decodeLogits, 0, vocabSize), vocabSize));
  }
}

}  // namespace llama
}  // namespace libllm
