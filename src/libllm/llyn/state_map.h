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

#include <string>
#include <unordered_map>
#include <utility>
#include "llyn/tensor.h"

namespace llyn {

// string -> Tensor dictioary. Usually used to store state-dict or kv-cache for a neural network.
class StateMap {
 public:
  StateMap() = default;
  ~StateMap();

  void read(const std::string &filename);

  // for tensors.
  Tensor getTensor(const std::string &name) const;
  void putTensor(const std::string &name, TensorCRef tensor);
  bool hasTensor(const std::string &name) const;

  // for numeric values. (currently, only int is supported)
  template<typename T>
  T getValue(const std::string &name) const;
  template<typename T>
  void putValue(const std::string &name, T value);
  template<typename T>
  bool hasValue(const std::string &name) const;

 private:
  std::unordered_map<std::string, Tensor> _dict;
  std::unordered_map<std::string, int> _intDict;

  std::pair<std::string, Tensor> readTensor(ly::ReadableFile *fp) const;
}; 

}  // namespace llyn
