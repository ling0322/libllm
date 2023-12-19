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

#include "ly/nn/test_helper.h"

#include "../../../third_party/catch2/catch_amalgamated.hpp"

namespace F = ly::functional;

namespace ly {
namespace nn {

ModuleTester::ModuleTester(Device device, DType weightType) :
    _device(device),
    _weightType(weightType),
    _random(RandomSeed) {}

Tensor ModuleTester::generateTensor(lut::Span<const Tensor::ShapeType> shape, float scale) {
  Tensor x = F::rand(shape, DType::kFloat, Device::getCpu(), &_random, -scale, scale);
  return toTargetDevice(x);
}

void ModuleTester::randomInit(std::shared_ptr<Module> module) {
  module->initParameters(&_random, _weightType);
}

Context ModuleTester::getCtx() const {
  Context ctx;
  ctx.setDevice(_device);
  ctx.setFloatDType(F::getDefaultFloatType(_device));

  return ctx;
}

Tensor ModuleTester::toTargetDevice(Tensor x) const {
  x = F::to(_device, x);
  if (x.getDType().isFloat()) {
    x = F::cast(x, F::getDefaultFloatType(_device));
  }

  return x;
}

Tensor ModuleTester::toCpu(Tensor x) const {
  if (x.getDType().isFloat()) {
    x = F::cast(x, F::getDefaultFloatType(Device::getCpu()));
  }
  x = F::to(Device::getCpu(), x);

  return x;
}

bool ModuleTester::allClose(Tensor a, lut::Span<const float> ref, float atol, float rtol) const {
  Tensor ar = Tensor::create<float>({static_cast<int>(ref.size())}, ref);
  bool allClose = F::allClose(a, ar, atol, rtol);
  if (!allClose) {
    LOG(ERROR) << "output tensor mismatch, expected:";
    F::print(ar);
    LOG(ERROR) << "but got:";
    F::print(a);
  }

  return allClose;
}

}  // namespace nn
}  // namespace ly
