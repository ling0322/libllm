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

#include "lten/operators.h"

#include <atomic>
#include <mutex>
#include <thread>

#include "lten/cpu/cpu_operators.h"
#include "lten/cpu/kernel/interface.h"
#include "lten/cuda/cuda_operators.h"
#include "lten/mp.h"
#include "lutil/error.h"
#include "lutil/strings.h"
#include "lutil/thread_pool.h"

namespace lten {

Tensor Operators::lookup(Tensor, Tensor) {
  NOT_IMPL();
}

Tensor Operators::matmul(Tensor, Tensor) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor, float) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor, Tensor) {
  NOT_IMPL();
}

Tensor Operators::softmax(Tensor) {
  NOT_IMPL();
}

Tensor Operators::sum(Tensor) {
  NOT_IMPL();
}

Tensor Operators::max(Tensor) {
  NOT_IMPL();
}

Tensor Operators::gelu(Tensor) {
  NOT_IMPL();
}

void Operators::fill(Tensor, float) {
  NOT_IMPL();
}

Tensor Operators::add(Tensor, Tensor) {
  NOT_IMPL();
}

Tensor Operators::tensor(lut::Span<const int>, DType) {
  NOT_IMPL();
}

Tensor Operators::tensorLike(Tensor) {
  NOT_IMPL();
}

Tensor Operators::zeros(lut::Span<const int>, DType) {
  NOT_IMPL();
}

bool Operators::allClose(Tensor, Tensor, float, float) {
  NOT_IMPL();
}

void Operators::print(Tensor) {
  NOT_IMPL();
}

Tensor Operators::layerNorm(Tensor, Tensor, Tensor, float) {
  NOT_IMPL();
}

Tensor Operators::rmsNorm(Tensor, Tensor, float) {
  NOT_IMPL();
}

Tensor Operators::causalMask(int) {
  NOT_IMPL();
}

Tensor Operators::applyRotaryPosEmb(Tensor, Tensor) {
  NOT_IMPL();
}

void Operators::copy(Tensor, Tensor) {
  NOT_IMPL();
}

Tensor Operators::swiglu(Tensor) {
  NOT_IMPL();
}

Tensor Operators::melFbank(Tensor) {
  NOT_IMPL();
}

Tensor Operators::to(Device, Tensor) {
  NOT_IMPL();
}

Tensor Operators::unfold(Tensor, int, int) {
  NOT_IMPL();
}

void Operators::repetitionPenalty(Tensor, Tensor, float) {
  NOT_IMPL();
}

Tensor Operators::cast(Tensor, DType) {
  NOT_IMPL();
}

DType Operators::getDefaultFloatType() {
  NOT_IMPL();
}

Tensor Operators::logMelSpectrogram(Tensor) {
  NOT_IMPL();
}

Tensor Operators::rand(lut::Span<const int>, DType, lut::Random *, float, float) {
  NOT_IMPL();
}

Operators *gOperatorsForDevice[Device::NumDeviceType] = {nullptr, nullptr};

static std::atomic<bool> gInitialized{false};

void initOperators() {
  op::cpu::kernel::init();

  if (!gInitialized.exchange(true)) {
    CHECK(!gOperatorsForDevice[Device::kCpu]);
    gOperatorsForDevice[Device::kCpu] = new op::cpu::CPUOperators();
#ifdef LIBLLM_CUDA_ENABLED
    CHECK(!gOperatorsForDevice[Device::kCuda]);
    gOperatorsForDevice[Device::kCuda] = llynCreateCudaOperators();
#endif

    MP::init();
  }
}

Operators *getOperators(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call getOperators() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    std::string deviceName = Device(deviceType).getName();
    throw lut::NotImplementedError(lut::sprintf("%s operators not implemented", deviceName));
  }

  return gOperatorsForDevice[deviceType];
}

bool isOperatorsAvailable(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call isOperatorsAvailable() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    return false;
  } else {
    return true;
  }
}

void destroyOperators() {
  op::cpu::kernel::destroy();
  MP::destroy();

  if (gInitialized.exchange(false)) {
    for (int i = 0; i < Device::NumDeviceType; ++i) {
      delete gOperatorsForDevice[i];
      gOperatorsForDevice[i] = nullptr;
    }
  }
}

}  // namespace lten
