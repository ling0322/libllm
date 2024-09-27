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

#include "libllm/tensor.h"
#include "lutil/reader.h"

namespace libllm {

// string -> Tensor dictioary. Usually used to store state-dict or kv-cache for a neural network.
class StateMap {
 public:
  StateMap() = default;
  ~StateMap();

  StateMap(StateMap &) = delete;
  StateMap &operator=(StateMap &) = delete;

  StateMap(StateMap &&rhs)
      : _dict(std::move(rhs._dict)),
        _intDict(std::move(rhs._intDict)) {
  }
  StateMap &operator=(StateMap &&rhs) {
    _dict = std::move(rhs._dict);
    _intDict = std::move(rhs._intDict);

    return *this;
  }

  /// @brief Create a copy of current state map. It's only a shallow copy and the content of Tensor
  /// will not be copied (original and copied Tensor still point to the same address of memory).
  /// @return Copied StateMap.
  StateMap clone() const;

  /// @brief Read the state map from stream.
  /// @param reader the stream.
  void read(lut::Reader *reader);

  // for tensors.
  Tensor getTensor(const std::string &name) const;
  void putTensor(const std::string &name, Tensor tensor);
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

  std::pair<std::string, Tensor> readTensor(lut::Reader *fp) const;
};

}  // namespace libllm
