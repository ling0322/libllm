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

#include "libllm/var_builder.h"

#include <utility>

#include "flint/functional.h"
#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"

namespace libllm {

VarBuilder::VarBuilder(const fl::Device &device, fl::DType floatType)
    : _device(device),
      _floatType(floatType) {
}

namespace {

std::pair<std::string, fl::Tensor> readTensor(lut::Reader *fp) {
  int16_t nameLen = fp->readValue<int16_t>();
  if (nameLen <= 0) {
    throw lut::AbortedError("invalid tensor map file (name_len)");
  }
  std::string name = fp->readString(nameLen);

  fl::Tensor tensor;
  tensor.read(fp);
  LOG(DEBUG) << lut::sprintf(
      "tensor %s: shape=%s, dtype=%s",
      name,
      tensor.getShapeString(),
      tensor.getDType().toString());

  return std::make_pair(name, tensor);
}

}  // namespace

// tensor_dict format
//   byte[4]: "TDIC"
//   int32_t: num_record
//   Record[num_record]:
//     int16_t: name_len
//     byte[name_len]: name
//     Tensor
//   int16_t: magic number 0x55aa
VarBuilder VarBuilder::fromStream(lut::Reader *fp, const fl::Device &device, fl::DType floatType) {
  std::string s = fp->readString(16);
  if (s != "llyn::tdicv2    ") {
    throw lut::AbortedError("unsupported tensor map file");
  }

  std::string tag = fp->readString(4);
  if (tag != "<d> ") throw lut::AbortedError("invalid tensor map file");
  tag = fp->readString(4);

  auto params = std::make_shared<std::unordered_map<std::string, fl::Tensor>>();
  while (tag != "</d>") {
    if (tag != "<r> ") throw lut::AbortedError("invalid tensor map file");

    std::pair<std::string, fl::Tensor> kv = readTensor(fp);
    LOG(DEBUG) << "Load tensor: " << kv.first;
    params->emplace(std::move(kv));

    tag = fp->readString(4);
    if (tag != "</r>") throw lut::AbortedError("invalid tensor map file");
    tag = fp->readString(4);
  }

  LOG(INFO) << params->size() << " tensors read.";

  VarBuilder vb(device, floatType);
  vb._params = params;

  return vb;
}

VarBuilder VarBuilder::withName(const std::string &name) const {
  CHECK(!name.empty());

  VarBuilder vb = *this;
  vb._ns = this->name(name);

  return vb;
}

fl::Tensor VarBuilder::get(const std::string &name, lut::Span<const int> shape) const {
  std::string fullName = this->name(name);

  auto it = _params->find(fullName);
  if (it == _params->end()) {
    throw lut::AbortedError(lut::sprintf("tensor \"%s\" not found in model.", fullName));
  }

  fl::Tensor tensor = it->second;
  tensor.throwIfInvalidShape(shape, fullName);

  tensor = fl::F::to(_device, tensor);
  if (tensor.getDType().isFloat()) tensor = fl::F::cast(tensor, _floatType);

  return tensor;
}

bool VarBuilder::has(const std::string &name) const {
  return _params->find(this->name(name)) != _params->end();
}

std::string VarBuilder::name(const std::string &name) const {
  if (_ns.empty()) return name;

  return _ns + "." + name;
}

std::string VarBuilder::name() const {
  return _ns;
}

const fl::Device &VarBuilder::getDevice() const {
  return _device;
}

fl::DType VarBuilder::getFloatDType() const {
  return _floatType;
}

}  // namespace libllm
