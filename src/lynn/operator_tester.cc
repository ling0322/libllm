// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "lynn/operator_tester.h"

#include <initializer_list>

#include "lutil/attributes.h"
#include "lutil/strings.h"
#include "lutil/time.h"
#include "lynn/device.h"
#include "lynn/functional.h"
#include "lynn/operators.h"

namespace ly {

OperatorTester::OperatorTester()
    : _printBenchmarkInfo(false),
      _atol(1e-6),
      _rtol(1e-5),
      _op(nullptr),
      _testDevice(Device::getCpu()),
      _testFloatType(DType::kFloat) {
}

OperatorTester OperatorTester::withOperators(Operators *operators) {
  OperatorTester tester = *this;
  tester._op = operators;
  tester._testFloatType = operators->getDefaultFloatType();
  return tester;
}

OperatorTester OperatorTester::withDevice(Device device) {
  OperatorTester tester = *this;
  tester._testDevice = device;
  return tester;
}

OperatorTester OperatorTester::withFloatType(DType dtype) {
  OperatorTester tester = *this;
  tester._testFloatType = dtype;
  return tester;
}

OperatorTester OperatorTester::withPrintBenchmarkInfo(bool isPrint) {
  OperatorTester tester = *this;
  tester._printBenchmarkInfo = isPrint;
  return tester;
}

OperatorTester OperatorTester::withTol(float rtol, float atol) {
  OperatorTester tester = *this;
  tester._rtol = rtol;
  tester._atol = atol;
  return tester;
}

bool OperatorTester::testToDevice(std::initializer_list<int> shape) {
  lut::Random random(MagicNumber);
  Tensor xr = F::rand(shape, DType::kFloat, Device::getCpu(), &random);
  Tensor x = _op->to(_testDevice, xr);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, _rtol);
}

bool OperatorTester::testCast(std::initializer_list<int> shape) {
  lut::Random random(MagicNumber);
  Tensor xr = F::rand(shape, DType::kFloat, Device::getCpu(), &random);
  Tensor x = _op->to(_testDevice, xr);
  x = _op->cast(x, _testFloatType);
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr);
}

bool OperatorTester::testCopy(std::initializer_list<int> shape, bool transpose) {
  lut::Random random(MagicNumber);
  Tensor tensor = F::rand(shape, DType::kFloat, Device::getCpu(), &random);
  Tensor x = _op->to(_testDevice, tensor);

  // cudaMemcpy path.
  x = _op->cast(x, _testFloatType);
  if (transpose) x = x.transpose(1, 0);
  Tensor x2 = _op->tensorLike(x);
  if (_printBenchmarkInfo) {
    LOG_TIME(_op->copy(x, x2), lut::sprintf("OP::copy(x, x2) t=%d", transpose));
  } else {
    _op->copy(x, x2);
  }

  x2 = _op->cast(x2, DType::kFloat);
  x2 = _op->to(Device::getCpu(), x2);
  if (transpose) x2 = x2.transpose(1, 0);
  return F::allClose(tensor, x2);
}

bool OperatorTester::testCopyLongType() {
  Tensor tensor = Tensor::create<LongType>({2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
  Tensor x = _op->to(_testDevice, tensor);

  // cudaMemcpy path (op::cuda::transform path is not supported for int16_t)
  Tensor x2 = _op->tensorLike(x);
  _op->copy(x, x2);
  x2 = _op->to(Device::getCpu(), x2);

  const LongType *px = tensor.getInternalData()->getData<LongType>(tensor.getInternalOffset()),
                 *pr = x2.getInternalData()->getData<LongType>(x2.getInternalOffset());
  x2.throwIfInvalidShape(tensor.getShape(), "OperatorTester::testCopyLongType");
  return std::equal(px, px + x2.getNumEl(), pr);
}

bool OperatorTester::testCopy5D() {
  lut::Random random(MagicNumber);
  Tensor tensor = F::rand({10, 2, 5, 20}, DType::kFloat, Device::getCpu(), &random);
  Tensor x = _op->to(_testDevice, tensor);

  x = _op->cast(x, _testFloatType);
  x = x.unsqueeze(1).expand({10, 4, 2, 5, 20});
  Tensor x2 = _op->tensorLike(x);
  _op->copy(x, x2);
  x = _op->cast(x2, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  Tensor xr = tensor.unsqueeze(1).expand({10, 4, 2, 5, 20});
  xr = F::contiguous(x);
  return F::allClose(x, xr);
}

bool OperatorTester::testLookup() {
  lut::Random random(MagicNumber);
  Tensor embd = F::rand({10, 32}, DType::kFloat, Device::getCpu(), &random);
  Tensor ids = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = _op->to(_testDevice, embd);
  x = _op->cast(x, _testFloatType);

  Tensor y = _op->to(_testDevice, ids);
  x = _op->lookup(x, y);
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  Tensor xr = F::lookup(embd, ids);
  return F::allClose(x, xr);
}

bool OperatorTester::testMatmul(ShapeType shapeA, ShapeType shapeB, bool transposeB) {
  lut::Random random(MagicNumber);
  Tensor a = F::rand(shapeA, DType::kFloat, Device::getCpu(), &random);
  Tensor b = F::rand(shapeB, DType::kFloat, Device::getCpu(), &random);
  Tensor xr = F::matmul(a, transposeB ? b.transpose(-1, -2) : b);

  Tensor x = _op->to(_testDevice, a);
  Tensor y = _op->to(_testDevice, b);
  x = _op->cast(x, _testFloatType);
  y = _op->cast(y, _testFloatType);
  if (transposeB) y = y.transpose(-1, -2);
  if (_printBenchmarkInfo) {
    LOG_TIME(x = _op->matmul(x, y), lut::sprintf("OP::matmul(x, x2) t=%d", transposeB));
  } else {
    x = _op->matmul(x, y);
  }
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, _rtol, _atol);
}

bool OperatorTester::testMatmulSlice(ShapeType shapeA, ShapeType shapeB) {
  lut::Random random(MagicNumber);
  Tensor a = F::rand(shapeA, DType::kFloat, Device::getCpu(), &random);
  Tensor b = F::rand(shapeB, DType::kFloat, Device::getCpu(), &random);
  Tensor xr = F::matmul(a, b.slice(-1, {8, 32}).transpose(-1, -2));

  Tensor x = _op->to(_testDevice, a);
  Tensor y = _op->to(_testDevice, b);
  x = _op->cast(x, _testFloatType);
  y = _op->cast(y, _testFloatType);
  y = y.slice(-1, {8, 32});
  y = y.transpose(-1, -2);
  x = _op->matmul(x, y);
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, _rtol, _atol);
}

bool OperatorTester::testMulScale() {
  lut::Random random(MagicNumber);
  Tensor a = F::rand({2, 5, 10}, DType::kFloat, Device::getCpu(), &random);
  Tensor xr = F::mul(a.transpose(2, 1).slice(1, {1, 9}), 0.1f);

  Tensor x = _op->to(_testDevice, a);
  x = _op->cast(x, _testFloatType);
  x = x.transpose(2, 1);
  x = x.slice(1, {1, 9});
  x = _op->mul(x, 0.1f);
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, 1e-3f, 1e-4f);
}

bool OperatorTester::testBinaryOp(OperatorTester::OperatorType op) {
  lut::Random random(MagicNumber);
  Tensor a = F::rand({2, 5, 10}, DType::kFloat, Device::getCpu(), &random);
  Tensor b = F::rand({5}, DType::kFloat, Device::getCpu(), &random);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xr;
  switch (op) {
    case OperatorType::Add:
      xr = F::add(at, b);
      break;
    case OperatorType::Sub:
      xr = F::sub(at, b);
      break;
    case OperatorType::Mul:
      xr = F::mul(at, b);
      break;
    default:
      NOT_IMPL();
  }

  Tensor x = _op->to(_testDevice, a);
  Tensor y = _op->to(_testDevice, b);
  x = _op->cast(x, _testFloatType);
  y = _op->cast(y, _testFloatType);
  x = x.transpose(2, 1);
  x = x.slice(1, {1, 9});
  switch (op) {
    case OperatorType::Add:
      x = _op->add(x, y);
      break;
    case OperatorType::Sub:
      x = _op->sub(x, y);
      break;
    case OperatorType::Mul:
      x = _op->mul(x, y);
      break;
    default:
      NOT_IMPL();
  }
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, _rtol, _atol);
}

bool OperatorTester::testUnaryOp(OperatorTester::OperatorType op, ShapeType shape) {
  lut::Random random(MagicNumber);
  Tensor a = F::rand(shape, DType::kFloat, Device::getCpu(), &random);
  Tensor xr;
  switch (op) {
    case OperatorType::Softmax:
      xr = F::softmax(a);
      break;
    case OperatorType::Swiglu:
      xr = F::swiglu(a);
      break;
    case OperatorType::Gelu:
      xr = F::gelu(a);
      break;
    default:
      NOT_IMPL();
  }

  Tensor x = _op->to(_testDevice, a);

  x = _op->cast(x, _testFloatType);
  if (op == OperatorType::Softmax && _printBenchmarkInfo) {
    LOG_TIME(x = _op->softmax(x), lut::sprintf("shape=%s OP::softmax(x)", a.getShapeString()));
  }
  if (op == OperatorType::Softmax && !_printBenchmarkInfo) x = _op->softmax(x);
  if (op == OperatorType::Swiglu) x = _op->swiglu(x);
  if (op == OperatorType::Gelu) x = _op->gelu(x);

  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, _rtol, _atol);
}

bool OperatorTester::testRmsNorm(ShapeType shape) {
  lut::Random random(MagicNumber);
  Tensor a = F::rand(shape, DType::kFloat, Device::kCpu, &random);
  Tensor b = F::rand({a.getShape(-1)}, DType::kFloat, Device::kCpu, &random);
  Tensor xr = F::rmsNorm(a, b, 1e-5);

  Tensor x = _op->to(_testDevice, a);
  Tensor y = _op->to(_testDevice, b);
  x = _op->cast(x, _testFloatType);
  y = _op->cast(y, _testFloatType);
  if (_printBenchmarkInfo) {
    LOG_TIME(
        x = _op->rmsNorm(x, y, 1e-5),
        lut::sprintf("shape=%s OP::rmsNorm", a.getShapeString()));
  } else {
    x = _op->rmsNorm(x, y, 1e-5);
  };
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, 5e-3f);
}

bool OperatorTester::testLayerNorm(ShapeType shape) {
  lut::Random random(MagicNumber);
  Tensor a = F::rand(shape, DType::kFloat, Device::kCpu, &random, -1, 100);
  Tensor mean = F::rand({a.getShape(-1)}, DType::kFloat, Device::kCpu, &random);
  Tensor var = F::rand({a.getShape(-1)}, DType::kFloat, Device::kCpu, &random);
  Tensor xr = F::layerNorm(a, mean, var, 1e-5);

  Tensor x = _op->to(_testDevice, a);
  x = _op->cast(x, _testFloatType);
  mean = _op->to(_testDevice, mean);
  mean = _op->cast(mean, _testFloatType);
  var = _op->to(_testDevice, var);
  var = _op->cast(var, _testFloatType);

  if (_printBenchmarkInfo) {
    LOG_TIME(
        x = _op->layerNorm(x, mean, var, 1e-5),
        lut::sprintf("shape=%s OP::layerNorm", a.getShapeString()));
  } else {
    x = _op->layerNorm(x, mean, var, 1e-5);
  };
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, _rtol, _atol);
}

bool OperatorTester::testCausalMask() {
  constexpr int DIM = 129;
  Tensor xr = F::softmax(F::causalMask(DIM));
  xr = F::cast(xr, DType::kFloat);

  Tensor x = _op->softmax(_op->causalMask(DIM));
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, 1e-3, 1e-4);
}

bool OperatorTester::testRoPE() {
  lut::Random random(MagicNumber);
  Tensor a = F::rand({2, 5, 2, 16}, DType::kFloat, Device::getCpu(), &random);
  Tensor b = F::rand({5, 1, 16}, DType::kFloat, Device::getCpu(), &random);
  Tensor xr = F::applyRotaryPosEmb(a, b);

  Tensor x = _op->to(_testDevice, a);
  Tensor y = _op->to(_testDevice, b);
  x = _op->cast(x, _testFloatType);
  y = _op->cast(y, _testFloatType);
  x = _op->applyRotaryPosEmb(x, y);
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, 5e-3f);
}

bool OperatorTester::testRepetitionPenalty() {
  lut::Random random(MagicNumber);
  Tensor a = F::rand({2, 16}, DType::kFloat, Device::getCpu(), &random);
  Tensor h = Tensor::create<LongType>({2, 4}, {1, 0, 1, 3, 0, 0, 0, 1});

  Tensor x = _op->to(_testDevice, a);
  Tensor y = _op->to(_testDevice, h);
  x = _op->cast(x, _testFloatType);
  _op->repetitionPenalty(x, y, 1.5);
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  F::repetitionPenalty(a, h, 1.5);
  return F::allClose(x, a, _rtol, _atol);
}

bool OperatorTester::testUnfold() {
  constexpr int DIM = 129;

  lut::Random random(MagicNumber);
  Tensor a = F::rand({2, 5, DIM}, DType::kFloat, Device::getCpu(), &random);
  Tensor xr = F::unfold(a, 5, 2);

  Tensor x = _op->to(_testDevice, a);
  x = _op->cast(x, _testFloatType);
  x = _op->unfold(x, 5, 2);
  x = _op->cast(x, DType::kFloat);
  x = _op->to(Device::getCpu(), x);

  return F::allClose(x, xr, _rtol, _atol);
}

}  // namespace ly
