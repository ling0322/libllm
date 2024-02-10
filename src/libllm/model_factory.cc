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

#include "libllm/model_factory.h"

#include "libllm/lut/error.h"
#include "libllm/lut/strings.h"
#include "libllm/constants.h"
#include "libllm/chatglm.h"
#include "libllm/llama.h"

namespace libllm {

std::shared_ptr<ModelForGeneration> ModelFactory::createModel(
    const Context &ctx,
    const lut::IniConfig &config) {
  std::string modelType = config.getSection(ModelSection).getString(ModelTypeField);

  LOG(INFO) << "model_type = " << modelType;
  LOG(INFO) << "device = " << ctx.getDevice().getName();

  if (modelType == "chatglm2" || modelType == "chatglm3")
    return chatglm::ChatGlmModelForGeneration::create(ctx, config);

  if (modelType == "llama")
    return llama::LlamaModelForGeneration::create(ctx, config);
  
  throw lut::AbortedError(lut::sprintf("unexpected model type: %s", modelType));
  return nullptr;
}

}  // namespace libllm