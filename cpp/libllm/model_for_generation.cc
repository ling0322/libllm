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

#include "libllm/bilibili_index.h"
#include "libllm/constants.h"
#include "libllm/llama.h"
#include "libllm/qwen.h"
#include "libllm/whisper.h"
#include "lutil/error.h"
#include "lutil/path.h"
#include "lutil/strings.h"
#include "lutil/zip_file.h"

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
  if (modelType == "llama") {
    model = llama::LlamaModelForGeneration::fromPackage(ctx, package);
  } else if (modelType == "index") {
    model = index::IndexModelForGeneration::fromPackage(ctx, package);
  } else if (modelType == "qwen") {
    model = qwen::QwenModelForGeneration::fromPackage(ctx, package);
  } else {
    throw lut::AbortedError(lut::sprintf("unexpected model type: %s", modelType));
  }

  return model;
}

void ModelForGeneration::initTokenizer(lut::ZipFile *package) {
  _tokenizer = Tokenizer::fromPackage(package);
}

const Vocab *ModelForGeneration::getVocab() const {
  return _tokenizer->getVocab();
}

void ModelForGeneration::encodePromptBlock(
    const PromptBlock &block,
    std::vector<LongType> &tokenIds) const {
  int tokenId;
  switch (block.blockType) {
    case PromptBlock::ControlToken:
      tokenId = _tokenizer->getVocab()->findControlToken(block.text);
      tokenIds.push_back(tokenId);
      LOG(DEBUG) << "control token " << block.text << " -> " << tokenId;
      break;
    case PromptBlock::Text:
      for (int tokenId : _tokenizer->encode(block.text)) {
        if (!getVocab()->isControlToken(tokenId)) {
          tokenIds.push_back(tokenId);
          LOG(DEBUG) << "token " << tokenId << " -> " << getVocab()->getTokenString(tokenId);
        }
      }
      break;
    default:
      NOT_IMPL();
  }
}

}  // namespace libllm
