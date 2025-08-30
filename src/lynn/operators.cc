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

#include "lynn/operators.h"

#include <atomic>
#include <mutex>
#include <thread>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "lutil/thread_pool.h"
#include "lynn/cpu/cpu_operators.h"
#include "lynn/cpu/kernel/interface.h"
#include "lynn/cuda/cuda_operators.h"
#include "lynn/mp.h"

namespace libllm {

Tensor Operators::arangeLong(LongType begin, LongType end, LongType step) {
  NOT_IMPL();
}

Tensor Operators::lookup(Tensor table, Tensor indices) {
  NOT_IMPL();
}

Tensor Operators::matmul(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor Operators::matmulNarrowPrecision(Tensor A, Tensor sfA, Tensor B, Tensor sfB) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor input, float other) {
  NOT_IMPL();
}

Tensor Operators::div(Tensor input, float other) {
  NOT_IMPL();
}

Tensor Operators::mod(Tensor input, LongType other) {
  NOT_IMPL();
}

Tensor Operators::eq(Tensor input, Tensor other) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor input, Tensor other) {
  NOT_IMPL();
}

Tensor Operators::softmax(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::sum(Tensor input, int dim) {
  NOT_IMPL();
}

Tensor Operators::max(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::gelu(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::square(Tensor input) {
  NOT_IMPL();
}

void Operators::fill(Tensor input, float value) {
  NOT_IMPL();
}

Tensor Operators::add(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor Operators::sub(Tensor a, Tensor b) {
  NOT_IMPL();
}

float Operators::elem(Tensor tensor) {
  NOT_IMPL();
}

bool Operators::elemBool(Tensor tensor) {
  NOT_IMPL();
}

Tensor Operators::tensor(lut::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor Operators::tensorLike(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::zeros(lut::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

bool Operators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  NOT_IMPL();
}

bool Operators::all(Tensor A) {
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

Tensor Operators::applyRotaryPosEmb(Tensor A, Tensor roPE) {
  NOT_IMPL();
}

void Operators::copy(Tensor src, Tensor dest) {
  NOT_IMPL();
}

Tensor Operators::swiglu(Tensor A) {
  NOT_IMPL();
}

Tensor Operators::melFbank(Tensor A) {
  NOT_IMPL();
}

Tensor Operators::to(Device device, Tensor tensor) {
  NOT_IMPL();
}

Tensor Operators::unfold(Tensor input, int kernelSize, int stride) {
  NOT_IMPL();
}

void Operators::repetitionPenalty(Tensor logits, Tensor history, float weight) {
  NOT_IMPL();
}

Tensor Operators::cast(Tensor tensor, DType dtype) {
  NOT_IMPL();
}

DType Operators::getDefaultFloatType() {
  NOT_IMPL();
}

Tensor Operators::logMelSpectrogram(Tensor wave) {
  NOT_IMPL();
}

void Operators::manualSeed(uint64_t seed) {
  NOT_IMPL();
}

Tensor Operators::rand(
    lut::Span<const int> shape,
    DType dtype,
    lut::Random *generator,
    float min,
    float max) {
  NOT_IMPL();
}

Tensor Operators::randNormal(lut::Span<const int> shape) {
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

}  // namespace libllm
