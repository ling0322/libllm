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
#include "flint/cpu/gated_delta_net.h"
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
#include "flint/cpu/unary.h"
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

Tensor CPUOperators::gatedDeltaNetPrefill(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor g,
    Tensor beta,
    Tensor cuSeqlens,
    Tensor stateSlots,
    Tensor state) {
  return cpu::gatedDeltaNetPrefill(q, k, v, g, beta, cuSeqlens, stateSlots, state);
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
    auto normalizedLogit = [&](int label) {
      float value = rowLogits[label];
      return std::isnan(value) ? -std::numeric_limits<float>::infinity() : value;
    };
    std::iota(labels.begin(), labels.end(), 0);
    std::sort(labels.begin(), labels.end(), [&](int left, int right) {
      return normalizedLogit(left) > normalizedLogit(right);
    });

    if (temperature == 0.0f) {
      sampled[row] = labels[0];
      continue;
    }

    int effectiveTopK = topK <= 0 ? vocabSize : topK;
    float maxLogit = normalizedLogit(labels[0]);
    auto samplingWeight = [&](int label) {
      float logit = normalizedLogit(label);
      if (std::isinf(maxLogit)) return logit == maxLogit ? 1.0f : 0.0f;
      return std::exp((logit - maxLogit) / temperature);
    };
    float totalWeight = 0.0f;
    for (int i = 0; i < effectiveTopK; ++i) {
      weights[i] = samplingWeight(labels[i]);
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

Tensor CPUOperators::min(Tensor inputs) {
  return cpu::reduce(inputs, MapReduceType::MIN);
}

Tensor CPUOperators::square(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::SQUARE);
}

Tensor CPUOperators::divTensor(Tensor input, Tensor other) {
  return cpu::binaryOp(input, other, BinaryOp::DIV);
}

Tensor CPUOperators::neg(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::NEG);
}

Tensor CPUOperators::abs(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::ABS);
}

Tensor CPUOperators::exp(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::EXP);
}

Tensor CPUOperators::sqrt(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::SQRT);
}

Tensor CPUOperators::rsqrt(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::RSQRT);
}

Tensor CPUOperators::sigmoid(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::SIGMOID);
}

Tensor CPUOperators::tanh(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::TANH);
}

Tensor CPUOperators::relu(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::RELU);
}

Tensor CPUOperators::gelu(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::GELU);
}

Tensor CPUOperators::silu(Tensor input) {
  return cpu::unaryOp(input, cpu::UnaryOp::SILU);
}

Tensor CPUOperators::div(Tensor input, float other) {
  return op::cpu::transform(input, 1.0f / other, 0.0f);
}

Tensor CPUOperators::arangeLong(LongType begin, LongType end, LongType step) {
  CHECK(step != 0);
  int64_t numel = (end - begin) / step;
  CHECK(numel >= 0 && numel < std::numeric_limits<int32_t>::max());

  Tensor x = op::cpu::tensor({static_cast<int>(numel)}, DType::kLong);
  LongType *data = getDataPtrCpu<LongType>(x);
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = begin + step * i;
  }

  return x;
}

Tensor CPUOperators::randNormal(lut::Span<const int> shape) {
  Tensor x = op::cpu::tensor(shape, DType::kFloat);
  int64_t numel = x.getNumEl();
  float *data = getDataPtrCpu<float>(x);

  // fillGaussian works in pairs, so an odd count is filled one element long and truncated.
  if (numel % 2 == 0) {
    _rand.fillGaussian(lut::Span<float>(data, numel));
  } else {
    std::vector<float> padded(numel + 1);
    _rand.fillGaussian(lut::makeSpan(padded));
    std::copy(padded.begin(), padded.begin() + numel, data);
  }

  return x;
}

float CPUOperators::elem(Tensor tensor) {
  CHECK(tensor.getNumEl() == 1);
  CHECK(tensor.getDType() == DType::kFloat);

  return getDataPtrCpu<float>(tensor)[0];
}

bool CPUOperators::elemBool(Tensor tensor) {
  CHECK(tensor.getNumEl() == 1);
  CHECK(tensor.getDType() == DType::kBool);

  return getDataPtrCpu<BoolType>(tensor)[0];
}

/// A packed copy of `input`, so the loops below can walk it as a flat array.
static Tensor contiguousCpu(const Tensor &input) {
  if (input.isContiguous()) return input;

  Tensor packed = op::cpu::tensorLike(input);
  op::cpu::copy(input, packed);
  return packed;
}

Tensor CPUOperators::mod(Tensor input, LongType other) {
  CHECK(input.getDType() == DType::kLong);
  CHECK(other != 0);

  Tensor x = contiguousCpu(input);
  Tensor c = op::cpu::tensorLike(x);
  const LongType *src = getDataPtrCpu<LongType>(x);
  LongType *dest = getDataPtrCpu<LongType>(c);
  for (int64_t i = 0; i < c.getNumEl(); ++i) {
    dest[i] = src[i] % other;
  }

  return c;
}

Tensor CPUOperators::eq(Tensor input, Tensor other) {
  // Matches the CUDA backend, which compares <uint8> tensors and answers in <bool>.
  CHECK(input.getDType() == DType::kUInt8 && other.getDType() == DType::kUInt8);
  input.throwIfInvalidShape(other.getShape(), "eq");

  Tensor a = contiguousCpu(input);
  Tensor b = contiguousCpu(other);
  Tensor c = op::cpu::tensor(a.getShape(), DType::kBool);

  const UInt8 *pa = getDataPtrCpu<UInt8>(a);
  const UInt8 *pb = getDataPtrCpu<UInt8>(b);
  BoolType *pc = getDataPtrCpu<BoolType>(c);
  for (int64_t i = 0; i < c.getNumEl(); ++i) {
    pc[i] = pa[i] == pb[i];
  }

  return c;
}

bool CPUOperators::all(Tensor A) {
  CHECK(A.getDType() == DType::kBool);

  Tensor x = contiguousCpu(A);
  const BoolType *data = getDataPtrCpu<BoolType>(x);
  for (int64_t i = 0; i < x.getNumEl(); ++i) {
    if (!data[i]) return false;
  }

  return true;
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
