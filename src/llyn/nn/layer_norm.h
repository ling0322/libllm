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
#include "llyn/context.h"
#include "llyn/nn/module.h"
#include "llyn/state_map.h"
#include "llyn/tensor.h"

namespace llyn {
namespace nn {

// layer-norm layer.
class LayerNorm : public Module {
 public:
  static std::unique_ptr<LayerNorm> create(const Context &ctx, int d_model, float eps = 1e-5);
  
  // implement interface nn::Module
  void initParameters(const StateMap &state_dict) override;

  // forward input and return the output.
  Tensor forward(const Tensor &input) const;
 
 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";
  static constexpr char kBias[] = "bias";

  Context _ctx;

  Tensor _w;
  Tensor _b;

  int _dModel;
  float _eps;

  LayerNorm();
};

}  // namespace nn
}  // namespace llyn
