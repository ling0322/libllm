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

#include "libllm/kv_cache.h"

#include "lutil/error.h"
#include "lutil/strings.h"

namespace libllm {

KVCache KVCache::clone() const {
  KVCache cache;
  cache._dict = _dict;
  cache._intDict = _intDict;

  return cache;
}

fl::Tensor KVCache::getTensor(const std::string &name) const {
  auto it = _dict.find(name);
  if (it == _dict.end()) {
    throw lut::AbortedError(lut::sprintf("tensor \"%s\" not found in kv cache.", name));
  }

  return it->second;
}

void KVCache::putTensor(const std::string &name, fl::Tensor tensor) {
  _dict[name] = tensor;
}

bool KVCache::hasTensor(const std::string &name) const {
  return _dict.find(name) != _dict.end();
}

int KVCache::getValue(const std::string &name) const {
  auto it = _intDict.find(name);
  if (it == _intDict.end()) {
    throw lut::AbortedError(lut::sprintf("value \"%s\" not found in kv cache.", name));
  }

  return it->second;
}

void KVCache::putValue(const std::string &name, int value) {
  _intDict[name] = value;
}

bool KVCache::hasValue(const std::string &name) const {
  return _intDict.find(name) != _intDict.end();
}

}  // namespace libllm
