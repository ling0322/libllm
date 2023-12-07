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

#include "llm/api/model_factory.h"

#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "llm/common/constants.h"
#include "llm/chatglm2/chatglm2_model_for_generation.h"
#include "llm/llama/llama_model_for_generation.h"

namespace libllm {

std::shared_ptr<ModelForGeneration> ModelFactory::createModel(
    const llyn::Context &ctx,
    const ly::IniConfig &config) {
  std::string modelType = config.getSection(ModelSection).getString(ModelTypeField);

  LOG(INFO) << "initialize " << modelType << " model, device = " << ctx.getDevice().getName();

  if (modelType == "chatglm2")
    return chatglm2::ChatGLM2ModelForGeneration::create(ctx, config);

  if (modelType == "llama")
    return llama::LlamaModelForGeneration::create(ctx, config);
  
  throw ly::AbortedError(ly::sprintf("unexpected model type: %s", modelType));
  return nullptr;
}

}  // namespace libllm
