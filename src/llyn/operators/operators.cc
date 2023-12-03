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

#include "llyn/operators/operators.h"

#include <atomic>
#include <mutex>
#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "llyn/operators/cpu/cpu_operators.h"
#include "llyn/operators/cuda/cuda_operators.h"

namespace llyn {
namespace internal {

Tensor Operators::lookup(Tensor table, Tensor indices) {
  NOT_IMPL();
}

Tensor Operators::matmul(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor input, float other) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor input, Tensor other) {
  NOT_IMPL();
}

Tensor Operators::softmax(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::gelu(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::add(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor Operators::createTensor(std::initializer_list<int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor Operators::createTensorLike(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::rand(std::initializer_list<int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor Operators::zeros(ly::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

bool Operators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  NOT_IMPL();
}

void Operators::print(Tensor tensor) {
  NOT_IMPL();
}

Tensor Operators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  NOT_IMPL();
}

Tensor Operators::rmsNorm(Tensor input, Tensor weight, float eps) {
  NOT_IMPL();
}

Tensor Operators::causalMask(int max_len) {
  NOT_IMPL();
}

Tensor Operators::cat(Tensor A, Tensor B, int dim) {
  NOT_IMPL();
}

Tensor Operators::applRotaryPosEmb(Tensor A, Tensor roPE) {
  NOT_IMPL();
}

void Operators::copy(Tensor src, Tensor dest) {
  NOT_IMPL();
}

Tensor Operators::attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
  NOT_IMPL();
}

Tensor Operators::swiglu(Tensor A) {
  NOT_IMPL();
}

Tensor Operators::toDevice(Tensor tensor, Device device) {
  NOT_IMPL();
}

Tensor Operators::cast(Tensor tensor, DType dtype) {
  NOT_IMPL();
}

DType Operators::getDefaultFloatType() {
  NOT_IMPL();
}

Operators *gOperatorsForDevice[Device::NumDeviceType] = {
  nullptr,
  nullptr
};

static std::atomic<bool> gInitialized{false};

void initOperators() {
  if (!gInitialized.exchange(true)) {
    CHECK(!gOperatorsForDevice[Device::kCpu]);
    gOperatorsForDevice[Device::kCpu] = new op::cpu::CPUOperators();

    CHECK(!gOperatorsForDevice[Device::kCuda]);
    gOperatorsForDevice[Device::kCuda] = llynCreateCudaOperators();
  }
}

Operators *getOperators(Device::Type deviceType) {
  if (!gInitialized) throw ly::AbortedError("call llyn operators before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    std::string deviceName = Device(deviceType).getName();
    throw ly::NotImplementedError(ly::sprintf("%s operators not implemented", deviceName));
  }

  return gOperatorsForDevice[deviceType];
}

void destroyOperators() {
  if (gInitialized.exchange(false)) {
    for (int i = 0; i < Device::NumDeviceType; ++i) {
      delete gOperatorsForDevice[i];
      gOperatorsForDevice[i] = nullptr;
    }
  }
}

}  // namespace internal
}  // namespace llyn
