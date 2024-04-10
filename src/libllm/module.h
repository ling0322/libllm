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
#include "libllm/context.h"
#include "libllm/state_map.h"
#include "libllm/tensor.h"
#include "libllm/lut/random.h"
#include "tensor.h"

namespace libllm {

// base class for all nn modules.
class Module {
 public:
  virtual ~Module() = default;

  // load the module states from `state_dict`
  virtual void initParameters(const StateMap &stateDict) = 0;

  /// @brief Initialize the module with random weights.
  /// @param generator Random number generator.
  /// @param quantType weight data type. For example, DType::kQ4 for Q4 quantization.
  virtual void initParameters(lut::Random *generator, DType weightType);

  /// @brief Move tensor to ctx.getDevice(), then if dtype of tensor is float, then cast it to
  ///        ctx.getFloatDType().
  /// @param tensor the input tensor.
  /// @param ctx Context for a module.
  /// @return tensor after move device and cast float.
  Tensor moveAndCastFloat(const Tensor &tensor, const Context &ctx);

  /// @brief Get context of current module.
  /// @return reference of Context.
  const Context &getCtx() const { return _ctx; }

  /// @brief Set the context of current module.
  /// @param ctx reference of Context.
  void setCtx(const Context &ctx) { _ctx = ctx; }

 private:
  Context _ctx;
};

// layer-norm layer.
class LayerNorm : public Module {
 public:
  static std::unique_ptr<LayerNorm> create(const Context &ctx, int d_model, float eps = 1e-5);
  
  // implement interface Module
  void initParameters(const StateMap &state_dict) override;

  // forward input and return the output.
  Tensor forward(const Tensor &input) const;
 
 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";
  static constexpr char kBias[] = "bias";

  Tensor _w;
  Tensor _b;

  int _dModel;
  float _eps;

  LayerNorm();
};

class RMSNorm : public Module {
 public:
  static constexpr char Weight[] = "weight";

  static std::unique_ptr<RMSNorm> create(const Context &ctx, int dModel, float eps);

  Tensor forward(const Tensor &input) const;

  // implement interface Module
  void initParameters(const StateMap &state_dict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

 private:
  Tensor _weight;

  int _dModel;
  float _eps;

  RMSNorm() = default;
};

class Embedding : public Module {
 public:
  static std::unique_ptr<Embedding> create(const Context &ctx, int dModel, int vocabSize);

  // implement interface Module
  void initParameters(const StateMap &state_dict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  // forward input and return the output.
  // Args:
  //   input <long>(N, L): input word-ids.
  // Returns:
  //   <float>(N, L, D): embeddings for input word-ids.
  Tensor forward(const Tensor &input) const;

 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";

  Tensor _wte;

  int _dModel;
  int _vocabSize;

  Embedding() = default;
};

class Linear : public Module {
 public:
  // create Linear module from context. 
  static std::unique_ptr<Linear> create(
      const Context &ctx, int inDim, int outDim, bool hasBias = true);

  // implement interface Module
  void initParameters(const StateMap &state_dict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  // forward input and return the output.
  Tensor forward(const Tensor &input) const;

 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";
  static constexpr char kBias[] = "bias";

  Tensor _w;
  Tensor _b;

  int _inDim;
  int _outDim;
  bool _hasBias;

  Linear();
};

}  // namespace libllm
