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

#pragma once

#include <memory>

#include "libllm/model_for_generation.h"
#include "libllm/module.h"
#include "lutil/ini_config.h"

namespace libllm {
namespace whisper {

struct WhisperConfig {
  int hiddenSize;
  int encoderNumHeads;
  int encoderFfnDim;
  int encoderNumLayers;
  int decoderNumLayers;
  int decoderFfnDim;
  int vocabSize;
  int maxTgtLength;

  WhisperConfig();

  static WhisperConfig loadConfig(const lut::IniSection &section);
};

class EncoderAttention : public Module {
 public:
  static std::shared_ptr<EncoderAttention> fromConfig(const Context &ctx, WhisperConfig config);
  ~EncoderAttention();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;
  Tensor forward(Tensor inputs);

 private:
  std::shared_ptr<Linear> _qkvProj;
  std::shared_ptr<Linear> _outProj;
  int _numHeads;
  int _hiddenSize;

  EncoderAttention();
};

class EncoderLayer : public Module {
 public:
  static std::shared_ptr<EncoderLayer> fromConfig(const Context &ctx, WhisperConfig config);
  ~EncoderLayer();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;
  Tensor forward(Tensor inputs);

 private:
  std::shared_ptr<LayerNorm> _norm1;
  std::shared_ptr<LayerNorm> _norm2;
  std::shared_ptr<EncoderAttention> _attn;
  std::shared_ptr<Linear> _fc1;
  std::shared_ptr<Linear> _fc2;

  EncoderLayer();
};

class EncoderModel : public Module {
 public:
  static std::shared_ptr<EncoderModel> fromConfig(const Context &ctx, WhisperConfig config);
  ~EncoderModel();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  /// @brief Forward the wave through the whisper encoder model and update the key-value cache in
  /// `past`.
  /// @param wave the input wave.
  Tensor forward(Tensor wave);

 private:
  static constexpr int FeatDim = 128;
  static constexpr int NumFrames = 30;
  static constexpr int InputSamples = 16000 * NumFrames;
  std::shared_ptr<Conv1D> _conv1;
  std::shared_ptr<Conv1D> _conv2;
  std::vector<std::shared_ptr<EncoderLayer>> _layers;
  std::shared_ptr<LayerNorm> _norm;
  Tensor _posEmbd;
  int _hiddenSize;

  EncoderModel();
};

class DecoderInitModel : public Module {
 public:
  static std::shared_ptr<DecoderInitModel> fromConfig(const Context &ctx, WhisperConfig config);
  ~DecoderInitModel();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  /// @brief Forward the encoderHidden through the cross attention kv-projection layers and update
  /// the key-value cache for cross attention in `past`.
  /// @param past the kv_cache to update.
  /// @param wave the hidden output from encoder model.
  void forward(StateMap &past, Tensor encoderHidden);

 private:
  std::vector<std::shared_ptr<Linear>> _kvProjs;
  int _dModel;

  DecoderInitModel();
};

class Attention : public Module {
 public:
  static std::shared_ptr<Attention> selfAttn(const Context &ctx, WhisperConfig config);
  static std::shared_ptr<Attention> crossAttn(const Context &ctx, WhisperConfig config);
  ~Attention();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;
  Tensor forward(StateMap &past, Tensor inputs);

 private:
  static constexpr int PastBlockSize = 2;

  std::shared_ptr<Linear> _proj;
  std::shared_ptr<Linear> _outProj;
  int _numHeads;
  int _hiddenSize;
  bool _selfAttn;

  std::string _namePastK;
  std::string _namePastV;
  std::string _namePastLen;

  Attention();

  /// @brief Common part of initialization for cross attention and self attention.
  /// @param config
  void initCommon(WhisperConfig config);

  /// @brief Get the present kv tensor from input kv and past kv tensors. NOTE: do not modify the
  /// content of returned tensors since they were the kv cache in next iteration.
  /// @param past the kv cache.
  /// @param k the input k.
  /// @param v the input v.
  std::pair<Tensor, Tensor> getPresentKV(StateMap &past, Tensor k, Tensor v);

  /// @brief Get context (history) length for the self attention.
  /// @param past the kv_cache.
  /// @return the context length.
  int getCtxLength(const StateMap &past) const;
};

class DecoderLayer : public Module {
 public:
  static constexpr char CrossAttn[] = "cross_attn";
  static constexpr char SelfAttn[] = "self_attn";

  static std::shared_ptr<DecoderLayer> fromConfig(const Context &ctx, WhisperConfig config);
  ~DecoderLayer();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  Tensor forward(StateMap &past, Tensor inputs);

 private:
  std::shared_ptr<LayerNorm> _norm1;
  std::shared_ptr<LayerNorm> _norm2;
  std::shared_ptr<LayerNorm> _norm3;
  std::shared_ptr<Attention> _selfAttn;
  std::shared_ptr<Attention> _crossAttn;
  std::shared_ptr<Linear> _fc1;
  std::shared_ptr<Linear> _fc2;

  DecoderLayer();
};

class DecoderModel : public Module {
 public:
  static std::shared_ptr<DecoderModel> fromConfig(const Context &ctx, WhisperConfig config);
  ~DecoderModel();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  Tensor forward(StateMap &past, Tensor inputs);
  Tensor forwardLmHead(Tensor inputs);
  int getOutputDim() const;

 private:
  std::vector<std::shared_ptr<DecoderLayer>> _layers;
  std::shared_ptr<Embedding> _embd;
  std::shared_ptr<LayerNorm> _norm;
  std::shared_ptr<Linear> _outProj;
  Tensor _posEmbd;
  std::string _namePastLen;
  int _dModel;
  int _maxTgtLength;
  int _outputDim;

  DecoderModel();

  /// @brief Get context (history) length for the positional embedding.
  /// @param past the kv_cache.
  /// @return the context length.
  int getCtxLength(const StateMap &past) const;
};

class WhisperLogitsProcessor : public LogitsProcessor {
 public:
  static std::shared_ptr<WhisperLogitsProcessor> newProcessor(const Vocab *vocab);

  void notifyToken(int tokenId) override;
  void processLogits(Tensor logits) override;

 private:
  static constexpr float Inf = std::numeric_limits<float>::infinity();

  std::vector<LongType> _history;

  int _lastTimeToken;
  int _beginTimeToken;
  int _endTimeToken;
  int _eotToken;
  int _transcribeToken;
  int _translateToken;
  int _noTimestampToken;
  int _langEnToken;
  int _langSuToken;

  int _lastTimeTokenIdx;

  WhisperLogitsProcessor();
};

class WhisperModelForGeneration : public ModelForGeneration {
 public:
  static std::shared_ptr<WhisperModelForGeneration> fromPackage(
      const Context &ctx,
      lut::ZipFile *package);

  Tensor prefill(StateMap &past, const Prompt &prompt) const override;
  Tensor decode(StateMap &past, LongType inputToken) const override;

  bool isStopToken(int tokenId) const override;
  const char *getName() const override;
  Device getDevice() const override;
  int getOutputDim() const override;

 protected:
  std::shared_ptr<EncoderModel> _encoder;
  std::shared_ptr<DecoderInitModel> _decoderInit;
  std::shared_ptr<DecoderModel> _decoder;
  std::string _modelName;
  int _eotId;

  WhisperModelForGeneration();
  void init(const Context &ctx, const lut::IniConfig &config);
  Tensor buildDecoderInput(lut::Span<const PromptBlock> prompt) const;
};

}  // namespace whisper
}  // namespace libllm
