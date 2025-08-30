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

#include <map>
#include <string>

#include "lynn/device.h"
#include "lynn/dtype.h"

namespace libllm {

// context for a module including operator set, device info and the namespace
class Context {
 public:
  static Context getCpu();

  // default constructor (root context).
  Context();

  // join two names or namespaces.
  static std::string joinName(const std::string &left, const std::string &right);

  // return a copy of this context with a new name under current context namespace.
  Context withName(const std::string &name) const;

  // get name under the namespace of this context. If no parameter given, return the name of the
  // context itself.
  std::string name(const std::string &name) const;
  std::string name() const {
    return _ns;
  }

  // device.
  const Device &getDevice() const;
  void setDevice(const Device &device) {
    _device = device;
  }

  // default float type.
  DType getFloatDType() const {
    return _floatType;
  }
  void setFloatDType(DType dtype) {
    _floatType = dtype;
  }

  /// Get or set value from the k-v store.
  std::string get(const std::string &key) const;
  void set(const std::string &key, const std::string &value);

  Context withDebugMode(bool debugMode) const;
  bool getDebugMode() const {
    return _debug;
  }

 private:
  std::string _ns;
  std::map<std::string, std::string> _propertyBag;
  Device _device;
  DType _floatType;
  bool _debug;
};

}  // namespace libllm
