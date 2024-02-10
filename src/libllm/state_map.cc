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

#include "libllm/state_map.h"

#include "libllm/functional.h"
#include "libllm/lut/error.h"
#include "libllm/lut/strings.h"
#include "libllm/lut/time.h"

namespace libllm {

StateMap::~StateMap() {}

// tensor_dict format
//   byte[4]: "TDIC"
//   int32_t: num_record
//   Record[num_record]:
//     int16_t: name_len
//     byte[name_len]: name
//     Tensor
//   int16_t: magic number 0x55aa
void StateMap::read(const std::string &filename) {
  _dict.clear();

  auto fp = lut::ReadableFile::open(filename);

  std::string s = fp->readString(16);
  if (s != "llyn::tdic      ") {
    throw lut::AbortedError("invalid tensor_dict file");
  }

  // place holder.
  fp->readValue<int32_t>();
  fp->readValue<int32_t>();

  // read tensors.
  int32_t numRecord = fp->readValue<int32_t>();
  double t0 = lut::now();
  LOG(INFO) << "read state map from " << filename;
  for (int i = 0; i < numRecord; ++i) {
    std::pair<std::string, Tensor> kv = readTensor(fp.get());
    if (lut::now() - t0 >= 5.0) {
      t0 = lut::now();
      LOG(INFO) << lut::sprintf("reading ... %.1f%%", 100.0 * i / numRecord);
    }

    _dict[kv.first] = kv.second;
  }
  LOG(INFO) << "reading ... 100.0%";
  LOG(INFO) << numRecord << " tensors read.";

  // magic number
  int16_t magicNumber = fp->readValue<int16_t>();
  if (magicNumber != 0x55aa) {
    throw lut::AbortedError("invalid tensor_dict file (magic number)");
  }
}

std::pair<std::string, Tensor> StateMap::readTensor(lut::ReadableFile *fp) const {
  int16_t nameLen = fp->readValue<int16_t>();
  if (nameLen <= 0) {
    throw lut::AbortedError("invalid tensor_dict file (name_len)");
  }
  std::string name = fp->readString(nameLen);
  LOG(DEBUG) << "read tensor " << name;

  Tensor tensor;
  tensor.read(fp);
  LOG(DEBUG) << lut::sprintf(
      "tensor %s: shape=%s, dtype=%s",
      name,
      tensor.getShapeString(),
      tensor.getDType().toString());  

  return std::make_pair(name, tensor);
}

Tensor StateMap::getTensor(const std::string &name) const {
  auto it = _dict.find(name);
  if (it == _dict.end()) {
    throw lut::AbortedError(lut::sprintf("tensor \"%s\" not found in state map.", name));
  }

  return it->second;
}

void StateMap::putTensor(const std::string &name, Tensor tensor) {
  _dict[name] = tensor;
}

bool StateMap::hasTensor(const std::string &name) const {
  return _dict.find(name) != _dict.end();
}

template<>
int StateMap::getValue<int>(const std::string &name) const {
  auto it = _intDict.find(name);
  if (it == _intDict.end()) {
    throw lut::AbortedError(lut::sprintf("value \"%s\" not found in state map.", name));
  }

  return it->second;
}

template<>
void StateMap::putValue<int>(const std::string &name, int value) {
  _intDict[name] = value;
}

template<>
bool StateMap::hasValue<int>(const std::string &name) const {
  return _intDict.find(name) != _intDict.end();
}

template int StateMap::getValue<int>(const std::string &name) const;
template void StateMap::putValue<int>(const std::string &name, int value);
template bool StateMap::hasValue<int>(const std::string &name) const;

}
