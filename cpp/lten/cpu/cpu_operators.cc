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

#include "lten/cpu/cpu_operators.h"

#include <stdlib.h>

#include <cmath>
#include <limits>
#include <memory>

#include "lten/cpu/all_close.h"
#include "lten/cpu/apply_rotary_pos_emb.h"
#include "lten/cpu/binary_op.h"
#include "lten/cpu/cast.h"
#include "lten/cpu/common.h"
#include "lten/cpu/copy.h"
#include "lten/cpu/cpu_tensor_data.h"
#include "lten/cpu/fill.h"
#include "lten/cpu/gelu.h"
#include "lten/cpu/kernel/interface.h"
#include "lten/cpu/log_mel_spectrogram.h"
#include "lten/cpu/lookup.h"
#include "lten/cpu/matmul.h"
#include "lten/cpu/normalizations.h"
#include "lten/cpu/print.h"
#include "lten/cpu/rand.h"
#include "lten/cpu/reduce.h"
#include "lten/cpu/repetition_penalty.h"
#include "lten/cpu/softmax.h"
#include "lten/cpu/swiglu.h"
#include "lten/cpu/tensor.h"
#include "lten/cpu/transform.h"
#include "lten/cpu/unfold.h"
#include "lten/operators.h"
#include "lten/tensor.h"

namespace lten {
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

Tensor CPUOperators::rand(
    lut::Span<const int> shape,
    DType dtype,
    lut::Random *generator,
    float min,
    float max) {
  return op::cpu::rand(shape, dtype, generator, min, max);
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

Tensor CPUOperators::softmax(Tensor input) {
  return cpu::softmax(input);
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

Tensor CPUOperators::gelu(Tensor input) {
  return cpu::gelu(input);
}

void CPUOperators::fill(Tensor input, float value) {
  return cpu::fill(input, value);
}

Tensor CPUOperators::sum(Tensor inputs) {
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

Tensor CPUOperators::applyRotaryPosEmb(Tensor A, Tensor roPE) {
  return cpu::applyRotaryPosEmb(A, roPE);
}

Tensor CPUOperators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  return cpu::layerNorm(input, weight, bias, eps);
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

Tensor CPUOperators::logMelSpectrogram(Tensor wave) {
  return cpu::logMelSpectrogram(wave);
}

Tensor CPUOperators::unfold(Tensor input, int kernelSize, int stride) {
  return cpu::unfold(input, kernelSize, stride);
}

Tensor CPUOperators::cast(Tensor tensor, DType dtype) {
  return cpu::cast(tensor, dtype);
}

DType CPUOperators::getDefaultFloatType() {
  return DType::getType<cpu::DefaultFloatType>();
}

}  // namespace cpu
}  // namespace op
}  // namespace lten
