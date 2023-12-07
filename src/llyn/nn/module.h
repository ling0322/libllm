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

#include "llyn/context.h"
#include "llyn/state_map.h"

namespace llyn {
namespace nn {

// base class for all nn modules.
class Module {
 public:
  virtual ~Module() = default;

  // load the module states from `state_dict`
  virtual void initParameters(const StateMap &stateDict) = 0;

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

}  // namespace nn
}  // namespace llyn
