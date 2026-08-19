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
#include <string>
#include <unordered_map>

#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/tensor.h"
#include "lutil/reader.h"
#include "lutil/span.h"

namespace libllm {

// reads the parameters of a model. A VarBuilder points to a namespace in the parameter file,
// tensors read from it are moved to the target device and float tensors are cast to the target
// float type.
class VarBuilder {
 public:
  // read the model parameters from `reader`. VarBuilder copies made by withName() share the same
  // parameters.
  static VarBuilder fromStream(lut::Reader *reader, const fl::Device &device, fl::DType floatType);

  // return a VarBuilder pointing to the sub-namespace `name` of this one.
  VarBuilder withName(const std::string &name) const;

  // return the tensor `name` in current namespace. Throws if it does not exist or its shape
  // mismatches `shape`.
  fl::Tensor get(const std::string &name, lut::Span<const int> shape) const;

  // same as get(), but for tensors whose shape is not known by the caller.
  fl::Tensor getUnchecked(const std::string &name) const;

  // return true if the tensor `name` exists in current namespace.
  bool has(const std::string &name) const;

  // get the full name of `name` in current namespace. If no parameter given, return the name of
  // the namespace itself.
  std::string name(const std::string &name) const;
  std::string name() const;

  const fl::Device &getDevice() const;
  fl::DType getFloatDType() const;

 private:
  std::shared_ptr<const std::unordered_map<std::string, fl::Tensor>> _params;
  std::string _ns;
  fl::Device _device;
  fl::DType _floatType;

  VarBuilder(const fl::Device &device, fl::DType floatType);
};

}  // namespace libllm
