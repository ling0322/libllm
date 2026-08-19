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

#include "flint/operators.h"

#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "lutil/thread_pool.h"
#include "flint/cpu/cpu_operators.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cuda/cuda_operators.h"

namespace fl {

namespace {

Tensor expandKeyValueHeads(Operators *op, Tensor input, int numHeads) {
  int batchSize = input.getShape(0);
  int numKeyValueHeads = input.getShape(1);
  int length = input.getShape(2);
  int headDim = input.getShape(3);
  int groupSize = numHeads / numKeyValueHeads;

  Tensor expanded =
      input.unsqueeze(2).expand({batchSize, numKeyValueHeads, groupSize, length, headDim});
  Tensor output = op->tensorLike(expanded);
  op->copy(expanded, output);

  return output.view({batchSize, numHeads, length, headDim});
}

}  // namespace

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

Tensor Operators::attention(Tensor q, Tensor k, Tensor v, bool causal) {
  CHECK(q.getDim() == 4 && k.getDim() == 4 && v.getDim() == 4);

  int numHeads = q.getShape(1);
  int numKeyValueHeads = k.getShape(1);
  int queryLength = q.getShape(2);
  int keyValueLength = k.getShape(2);
  int headDim = q.getShape(3);
  CHECK(numHeads % numKeyValueHeads == 0);

  if (numHeads != numKeyValueHeads) {
    k = expandKeyValueHeads(this, k, numHeads);
    v = expandKeyValueHeads(this, v, numHeads);
  }

  // Scaling both q and k keeps the scores in range for half precision.
  float scale = sqrtf(1.0f / sqrtf(1.0f * headDim));
  Tensor scores = matmul(mul(q, scale), mul(k, scale).transpose(-2, -1));

  // A single query attends to the whole history, so it needs no mask.
  if (causal && queryLength > 1) {
    Tensor mask =
        causalMask(keyValueLength).slice(0, {keyValueLength - queryLength, keyValueLength});
    scores = add(scores, mask);
  }

  return matmul(softmax(scores), v);
}

Tensor Operators::sum(Tensor input, int dim) {
  NOT_IMPL();
}

Tensor Operators::max(Tensor input) {
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

Tensor Operators::subFloat(Tensor input, float other) {
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

Tensor Operators::rmsNorm(Tensor input, Tensor weight, float eps) {
  NOT_IMPL();
}

Tensor Operators::sample(Tensor distribution, int topK, float topP) {
  NOT_IMPL();
}

Tensor Operators::causalMask(int max_len) {
  NOT_IMPL();
}

void Operators::copy(Tensor src, Tensor dest) {
  NOT_IMPL();
}

Tensor Operators::swiglu(Tensor A) {
  NOT_IMPL();
}

Tensor Operators::to(Device device, Tensor tensor) {
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

void Operators::manualSeed(uint64_t seed) {
  NOT_IMPL();
}

Tensor Operators::rand(lut::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor Operators::randNormal(lut::Span<const int> shape) {
  NOT_IMPL();
}

std::shared_ptr<Operators> gOperatorsForDevice[Device::NumDeviceType] = {nullptr, nullptr};

static std::atomic<bool> gInitialized{false};

void initOperators() {
  op::cpu::kernel::init();

#ifdef _OPENMP
  LOG(INFO) << "OMP max_threads = " << omp_get_max_threads();
#endif

  if (!gInitialized.exchange(true)) {
    CHECK(!gOperatorsForDevice[Device::kCpu]);
    gOperatorsForDevice[Device::kCpu] = std::make_shared<op::cpu::CPUOperators>();
#ifdef LIBLLM_CUDA_ENABLED
    CHECK(!gOperatorsForDevice[Device::kCuda]);
    gOperatorsForDevice[Device::kCuda] = op::cuda::CudaOperators::create();
#endif
  }
}

Operators *getOperators(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call getOperators() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    std::string deviceName = Device(deviceType).getName();
    throw lut::NotImplementedError(lut::sprintf("%s operators not implemented", deviceName));
  }

  return gOperatorsForDevice[deviceType].get();
}

std::shared_ptr<Operators> getOperatorsSharedPtr(Device::Type deviceType) {
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

  if (gInitialized.exchange(false)) {
    for (int i = 0; i < Device::NumDeviceType; ++i) {
      gOperatorsForDevice[i] = nullptr;
    }
  }
}

}  // namespace fl
