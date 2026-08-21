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

#include "flint/cpu/cpu_operators.h"

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

#include "lutil/random.h"
#include "flint/cpu/all_close.h"
#include "flint/cpu/binary_op.h"
#include "flint/cpu/cast.h"
#include "flint/cpu/common.h"
#include "flint/cpu/copy.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/cpu/fill.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cpu/lookup.h"
#include "flint/cpu/matmul.h"
#include "flint/cpu/normalizations.h"
#include "flint/cpu/print.h"
#include "flint/cpu/rand.h"
#include "flint/cpu/reduce.h"
#include "flint/cpu/repetition_penalty.h"
#include "flint/cpu/softmax.h"
#include "flint/cpu/swiglu.h"
#include "flint/cpu/tensor.h"
#include "flint/cpu/transform.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

CPUOperators::CPUOperators() {
}

Tensor CPUOperators::tensor(lut::Span<const int> shape, DType dtype) {
  return op::cpu::tensor(shape, dtype);
}

Tensor CPUOperators::tensorLike(Tensor input) {
  return op::cpu::tensorLike(input);
}

// -- class CPUOperators ----------

Tensor CPUOperators::rand(lut::Span<const int> shape, DType dtype) {
  return op::cpu::rand(shape, dtype, &_rand, 0, 1);
}

Tensor CPUOperators::zeros(lut::Span<const int> shape, DType dtype) {
  return op::cpu::zeros(shape, dtype);
}

Tensor CPUOperators::matmul(Tensor A, Tensor B) {
  return cpu::matmul(A, B);
}

void CPUOperators::print(Tensor tensor) {
  return cpu::print(tensor);
}

Tensor CPUOperators::add(Tensor input, Tensor other) {
  return cpu::binaryOp(input, other, BinaryOp::ADD);
}

Tensor CPUOperators::sub(Tensor input, Tensor other) {
  return cpu::binaryOp(input, other, BinaryOp::SUB);
}

Tensor CPUOperators::subFloat(Tensor input, float other) {
  return op::cpu::transform(input, 1.0f, -other);
}

Tensor CPUOperators::softmax(Tensor input) {
  return cpu::softmax(input);
}

Tensor CPUOperators::sample(
    Tensor logits,
    Tensor temperatures,
    Tensor topKs,
    Tensor topPs) {
  CHECK(logits.getDevice().getType() == Device::kCpu && logits.getDim() == 2);
  CHECK(logits.isContiguous());
  int rows = logits.getShape(0);
  int vocabSize = logits.getShape(1);
  CHECK(temperatures.getDevice().getType() == Device::kCpu &&
        temperatures.getDType() == DType::kFloat && temperatures.isContiguous() &&
        temperatures.getShape() == std::vector<int>{rows});
  CHECK(topKs.getDevice().getType() == Device::kCpu &&
        topKs.getDType() == DType::kInt32 && topKs.isContiguous() &&
        topKs.getShape() == std::vector<int>{rows});
  CHECK(topPs.getDevice().getType() == Device::kCpu &&
        topPs.getDType() == DType::kFloat && topPs.isContiguous() &&
        topPs.getShape() == std::vector<int>{rows});

  if (logits.getDType() == DType::kFloat16) {
    return sample(cast(logits, DType::kFloat), temperatures, topKs, topPs);
  }
  CHECK(logits.getDType() == DType::kFloat);

  const float *logitData = getDataPtrCpu<float>(logits);
  const float *temperatureData = getDataPtrCpu<float>(temperatures);
  const IntType *topKData = getDataPtrCpu<IntType>(topKs);
  const float *topPData = getDataPtrCpu<float>(topPs);
  std::vector<LongType> sampled(rows);
  std::vector<int> labels(vocabSize);
  std::vector<float> weights(vocabSize);

  for (int row = 0; row < rows; ++row) {
    float temperature = temperatureData[row];
    int topK = topKData[row];
    float topP = topPData[row];
    CHECK(std::isfinite(temperature) && temperature >= 0.0f);
    CHECK(topK >= -1 && topK <= vocabSize);
    CHECK(topP > 0.0f && topP <= 1.0f);

    const float *rowLogits = logitData + static_cast<int64_t>(row) * vocabSize;
    std::iota(labels.begin(), labels.end(), 0);
    std::sort(labels.begin(), labels.end(), [&](int left, int right) {
      return rowLogits[left] > rowLogits[right];
    });

    if (temperature == 0.0f) {
      sampled[row] = labels[0];
      continue;
    }

    int effectiveTopK = topK <= 0 ? vocabSize : topK;
    float maxLogit = rowLogits[labels[0]];
    float totalWeight = 0.0f;
    for (int i = 0; i < effectiveTopK; ++i) {
      weights[i] = std::exp((rowLogits[labels[i]] - maxLogit) / temperature);
      totalWeight += weights[i];
    }

    float selectedWeight = 0.0f;
    int selectedCount = effectiveTopK;
    for (int i = 0; i < effectiveTopK; ++i) {
      selectedWeight += weights[i];
      if (selectedWeight >= topP * totalWeight) {
        selectedCount = i + 1;
        break;
      }
    }

    float draw = _rand.nextFloat() * selectedWeight;
    float cumulative = 0.0f;
    sampled[row] = labels[selectedCount - 1];
    for (int i = 0; i < selectedCount; ++i) {
      cumulative += weights[i];
      if (draw < cumulative) {
        sampled[row] = labels[i];
        break;
      }
    }
  }

  return Tensor::create<LongType>({rows}, sampled);
}

bool CPUOperators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  return cpu::allClose(A, B, rtol, atol);
}

Tensor CPUOperators::mul(Tensor A, float k) {
  return op::cpu::transform(A, k, 0.0f);
}

Tensor CPUOperators::mul(Tensor A, Tensor B) {
  return op::cpu::binaryOp(A, B, BinaryOp::MUL);
}

Tensor CPUOperators::lookup(Tensor table, Tensor indices) {
  return cpu::lookup(table, indices);
}

void CPUOperators::fill(Tensor input, float value) {
  return cpu::fill(input, value);
}

Tensor CPUOperators::sum(Tensor inputs, int dim) {
  CHECK(dim == -1 || dim == inputs.getDim() - 1);
  return cpu::reduce(inputs, MapReduceType::SUM);
}

Tensor CPUOperators::max(Tensor inputs) {
  return cpu::reduce(inputs, MapReduceType::MAX);
}

void CPUOperators::repetitionPenalty(Tensor logits, Tensor history, float weight) {
  CHECK(history.getDType() == DType::kLong);

  return cpu::repetitionPenalty(logits, history, weight);
}

Tensor CPUOperators::rmsNorm(Tensor input, Tensor weight, float eps) {
  CHECK(input.getDType() == weight.getDType());

  return cpu::rmsNorm(input, weight, eps);
}

Tensor CPUOperators::causalMask(int max_len) {
  return op::cpu::causalMask(max_len, getDefaultFloatType());
}

void CPUOperators::copy(Tensor src, Tensor dest) {
  return cpu::copy(src, dest);
}

Tensor CPUOperators::swiglu(Tensor A) {
  return cpu::swiglu(A);
}

Tensor CPUOperators::to(Device device, Tensor tensor) {
  if (device.getType() == Device::kCpu) return tensor;

  NOT_IMPL();
}

Tensor CPUOperators::cast(Tensor tensor, DType dtype) {
  return cpu::cast(tensor, dtype);
}

void CPUOperators::manualSeed(uint64_t seed) {
  _rand.reset(seed);
}

DType CPUOperators::getDefaultFloatType() {
  return DType::getType<cpu::DefaultFloatType>();
}

// the CPU backend allocates through the system allocator and tracks nothing.
MemorySnapshot CPUOperators::captureMemorySnapshot() {
  return MemorySnapshot(0, 0, 0, 0);
}

void CPUOperators::resetPeakMemoryStats() {
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
