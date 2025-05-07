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

#include "libllm/context.h"

#include "libllm/device.h"
#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"

namespace libllm {

Context Context::getCpu() {
  Context ctx;
  ctx.setDevice(Device::getCpu());

  return ctx;
}

Context::Context()
    : _floatType(DType::kFloat),
      _debug(false) {
}

Context Context::withName(const std::string &name) const {
  CHECK(!name.empty());
  Context ctx = *this;
  ctx._ns = this->name(name);

  return ctx;
}

Context Context::withDebugMode(bool debugMode) const {
  Context ctx = *this;
  ctx._debug = debugMode;

  return ctx;
}

const Device &Context::getDevice() const {
  return _device;
}

std::string Context::joinName(const std::string &left, const std::string &right) {
  std::string join = left;
  join += ".";
  join += right;
  return join;
}

std::string Context::name(const std::string &name) const {
  std::string ns = _ns;

  if (ns.empty()) {
    ns = name;
  } else {
    ns = joinName(ns, name);
  }

  return ns;
}

std::string Context::get(const std::string &key) const {
  if (_propertyBag.find(key) == _propertyBag.end()) {
    THROW(Aborted, lut::sprintf("key not exist: \"%s\"", key));
  }

  return _propertyBag.at(key);
}

void Context::set(const std::string &key, const std::string &value) {
  _propertyBag[key] = value;
}

}  // namespace libllm
