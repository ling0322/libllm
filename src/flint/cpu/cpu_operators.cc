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

#include <cmath>
#include <limits>
#include <memory>

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

}  // namespace cpu
}  // namespace op
}  // namespace fl
