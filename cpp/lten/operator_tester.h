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

#pragma once

#include "lten/device.h"
#include "lten/operators.h"
#include "lutil/attributes.h"

namespace lten {

class OperatorTester {
 public:
  using ShapeType = std::initializer_list<int>;

  static constexpr uint32_t MagicNumber = 0x33;
  enum class OperatorType { Add, Mul, Softmax, Swiglu, Gelu };

  OperatorTester();

  OperatorTester withOperators(Operators *operators);
  OperatorTester withDevice(Device device);
  OperatorTester withFloatType(DType dtype);
  OperatorTester withPrintBenchmarkInfo(bool isPrint);
  OperatorTester withTol(float rtol = 1e-4, float atol = 1e-5);

  LUT_CHECK_RETURN bool testToDevice(std::initializer_list<int> shape);
  LUT_CHECK_RETURN bool testCast(std::initializer_list<int> shape);
  LUT_CHECK_RETURN bool testCopy(std::initializer_list<int> shape, bool transpose);
  LUT_CHECK_RETURN bool testCopyLongType();
  LUT_CHECK_RETURN bool testCopy5D();
  LUT_CHECK_RETURN bool testLookup();
  LUT_CHECK_RETURN bool testLookupQInt4();
  LUT_CHECK_RETURN bool testMatmul(ShapeType shapeA, ShapeType shapeB, bool transposeB);
  LUT_CHECK_RETURN bool testMatmulSlice(ShapeType shapeA, ShapeType shapeB);
  LUT_CHECK_RETURN bool testMatmulQInt4(ShapeType shapeA, ShapeType shapeB, bool transposeB);
  LUT_CHECK_RETURN bool testMulScale();
  LUT_CHECK_RETURN bool testBinaryOp(OperatorType op);
  LUT_CHECK_RETURN bool testUnaryOp(OperatorType op, ShapeType shape);
  LUT_CHECK_RETURN bool testRmsNorm(ShapeType shape);
  LUT_CHECK_RETURN bool testLayerNorm(ShapeType shape);
  LUT_CHECK_RETURN bool testCausalMask();
  LUT_CHECK_RETURN bool testRoPE();
  LUT_CHECK_RETURN bool testUnfold();
  LUT_CHECK_RETURN bool testRepetitionPenalty();

 private:
  bool _printBenchmarkInfo;
  float _atol;
  float _rtol;
  Operators *_op;
  Operators *_referenceOp;

  Device _testDevice;
  DType _testFloatType;
};

}  // namespace lten
