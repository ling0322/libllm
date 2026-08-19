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

#pragma once

#include <memory>

#include "libllm/constants.h"
#include "libllm/kv_cache.h"
#include "libllm/layers.h"
#include "libllm/model_for_generation.h"
#include "libllm/packed_batch.h"
#include "lutil/error.h"
#include "lutil/ini_config.h"
#include "flint/functional.h"

namespace libllm {
namespace llama {

struct LlamaConfig {
  int hiddenSize;
  int numHeads;
  int numKeyValueHeads;
  int intermediateSize;
  float normEps;
  int numLayers;
  int vocabSize;
  int maxContextLength;
  bool qkvProjBias;

  LlamaConfig();

  static LlamaConfig loadConfig(const lut::IniSection &section);
};

class MLP {
 public:
  static std::shared_ptr<MLP> build(const LlamaConfig &config, const VarBuilder &vb);

  fl::Tensor forward(fl::Tensor input) const;

 private:
  std::shared_ptr<Linear> _gateUpProj;
  std::shared_ptr<Linear> _downProj;

  MLP() = default;
};

class Attention {
 public:
  // `roPE` is shared by all the attention layers of a model.
  static std::shared_ptr<Attention> build(
      const LlamaConfig &config,
      const VarBuilder &vb,
      fl::Tensor roPE);

  // `batch` describes how `input` decomposes into sequences. For today's callers it is always
  // `PackedBatch::single(...)`, i.e. a single contiguous sequence; multi-sequence packing is
  // wired up in a later phase.
  fl::Tensor forward(KVCache &past, fl::Tensor input, const PackedBatch &batch) const;

  // get past length of this attention layer. Exposed so callers building a `PackedBatch` for the
  // single-sequence case know the correct starting rotary position.
  int getCtxLength(const KVCache &past) const;

 private:
  std::shared_ptr<Linear> _qkvProj;
  std::shared_ptr<Linear> _outProj;
  fl::Tensor _roPE;

  std::string _namePastK;
  std::string _namePastV;
  std::string _namePastLen;

  int _hiddenSize;
  int _numHead;
  int _numKeyValueHead;
  int _headDim;

  Attention();

  fl::Tensor applyRoPE(fl::Tensor x, fl::Tensor roPE) const;
  fl::Tensor rotateHalf(fl::Tensor x) const;
};

class DecodeLayer {
 public:
  static std::shared_ptr<DecodeLayer> build(
      const LlamaConfig &config,
      const VarBuilder &vb,
      fl::Tensor roPE);

  fl::Tensor forward(KVCache &past, fl::Tensor input, const PackedBatch &batch) const;
  int getCtxLength(const KVCache &past) const;

 private:
  std::shared_ptr<RMSNorm> _inputNorm;
  std::shared_ptr<RMSNorm> _postAttnNorm;
  std::shared_ptr<Attention> _attn;
  std::shared_ptr<MLP> _mlp;

  DecodeLayer() = default;
};

class LlamaModel {
 public:
  static std::shared_ptr<LlamaModel> build(const LlamaConfig &config, const VarBuilder &vb);

  // convenience overload for the single-sequence case (every caller today): builds a
  // `PackedBatch::single(...)` internally from `past`'s current length and `input`'s length.
  fl::Tensor forward(KVCache &past, fl::Tensor input) const;
  fl::Tensor forward(KVCache &past, fl::Tensor input, const PackedBatch &batch) const;
  fl::Tensor forwardLmHead(fl::Tensor hidden) const;
  int getOutputDim() const;

 private:
  LlamaConfig _config;
  std::shared_ptr<Embedding> _embedding;
  std::shared_ptr<RMSNorm> _norm;
  std::vector<std::shared_ptr<DecodeLayer>> _layers;
  std::shared_ptr<Linear> _outProj;

  LlamaModel() = default;
};

class LlamaModelForGeneration : public ModelForGeneration {
 public:
  static std::shared_ptr<LlamaModelForGeneration> fromPackage(
      const fl::Device &device,
      lut::ZipFile *package);

  fl::Tensor prefill(KVCache &past, const Prompt &prompt) const override;
  fl::Tensor decode(KVCache &past, fl::LongType inputToken) const override;

  bool isStopToken(int tokenId) const override;
  const char *getName() const override;
  fl::Device getDevice() const override;
  int getOutputDim() const override;
  Prompt buildPrompt(lut::Span<const Message> history) const override;

 protected:
  std::shared_ptr<LlamaModel> _model;
  std::string _modelName;
  fl::Device _device;
  int _eotId;

  LlamaModelForGeneration();
  fl::Tensor buildInput(const Prompt &prompt) const;
};

}  // namespace llama
}  // namespace libllm