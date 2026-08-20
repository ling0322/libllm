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

#include "flint/tensor.h"
#include "libllm/var_builder.h"

namespace libllm {

class RMSNorm {
 public:
  static constexpr char Weight[] = "weight";

  static std::unique_ptr<RMSNorm> build(int dModel, float eps, const VarBuilder &vb);

  fl::Tensor forward(const fl::Tensor &input) const;

 private:
  fl::Tensor _weight;
  float _eps;

  RMSNorm() = default;
};

class Embedding {
 public:
  static std::unique_ptr<Embedding> build(int dModel, int vocabSize, const VarBuilder &vb);

  // forward input and return the output.
  // Args:
  //   input <long>(L): packed input word-ids.
  // Returns:
  //   <float>(L, D): embeddings for input word-ids.
  fl::Tensor forward(const fl::Tensor &input) const;

 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";

  fl::Tensor _wte;

  Embedding() = default;
};

class Linear {
 public:
  static std::unique_ptr<Linear> build(
      int inDim,
      int outDim,
      bool hasBias,
      const VarBuilder &vb);

  // forward input and return the output.
  fl::Tensor forward(const fl::Tensor &input) const;

 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";
  static constexpr char kBias[] = "bias";

  fl::Tensor _w;
  fl::Tensor _b;

  bool _hasBias;

  Linear();
};

}  // namespace libllm
