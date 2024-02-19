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

#include "libllm/model_for_generation.h"

#include "libllm/lut/error.h"
#include "libllm/lut/path.h"
#include "libllm/lut/strings.h"
#include "libllm/lut/zip_file.h"
#include "libllm/constants.h"
#include "libllm/chatglm.h"
#include "libllm/llama.h"
#include "libllm/qwen.h"

namespace libllm {

constexpr char ModelForGeneration::ModelConfig[];

std::shared_ptr<ModelForGeneration> ModelForGeneration::fromPackage(
    const Context &fromCtx,
    lut::ZipFile *package) {
  std::shared_ptr<lut::IniConfig> ini = lut::IniConfig::fromStream(
      package->open(ModelConfig).get());

  std::string modelType = ini->getSection(ModelSection).getString(ModelTypeField);

  LOG(INFO) << "model_type = " << modelType;
  LOG(INFO) << "device = " << fromCtx.getDevice().getName();

  Context ctx = fromCtx.withName(modelType);
  std::shared_ptr<ModelForGeneration> model;
  if (modelType == "chatglm2" || modelType == "chatglm3") {
    model = chatglm::ChatGlmModelForGeneration::fromConfig(ctx, *ini);
  } else if (modelType == "llama") {
    model =  llama::LlamaModelForGeneration::fromConfig(ctx, *ini);
  } else if (modelType == "qwen") {
    model =  qwen::QwenModelForGeneration::fromConfig(ctx, *ini);
  } else {
    throw lut::AbortedError(lut::sprintf("unexpected model type: %s", modelType));
  }

  // read state map.
  std::string modelFile = ini->getSection(ModelSection).getString(ModelFileField);

  StateMap stateMap;
  stateMap.read(package->open(modelFile).get());

  // initialize parameters.
  model->initParameters(stateMap);
  return model;
}

}  // namespace libllm
