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

#include "flint/tensor.h"

namespace libllm {

// the past key-value tensors of a generation session, as well as the past length of each attention
// layer.
class KVCache {
 public:
  KVCache() = default;

  KVCache(const KVCache &) = delete;
  KVCache &operator=(const KVCache &) = delete;

  KVCache(KVCache &&rhs) = default;
  KVCache &operator=(KVCache &&rhs) = default;

  /// @brief Create a copy of current cache. It's only a shallow copy and the content of Tensor
  /// will not be copied (original and copied Tensor still point to the same address of memory).
  /// @return Copied KVCache.
  KVCache clone() const;

  // for tensors.
  fl::Tensor getTensor(const std::string &name) const;
  void putTensor(const std::string &name, fl::Tensor tensor);
  bool hasTensor(const std::string &name) const;

  // for lengths.
  int getValue(const std::string &name) const;
  void putValue(const std::string &name, int value);
  bool hasValue(const std::string &name) const;

 private:
  std::unordered_map<std::string, fl::Tensor> _dict;
  std::unordered_map<std::string, int> _intDict;
};

}  // namespace libllm
