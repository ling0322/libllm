// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lutil/attributes.h"
#include "lynn/device.h"
#include "lynn/operators.h"

namespace ly {

class OperatorBenchmark {
 public:
  OperatorBenchmark();

  OperatorBenchmark withOperators(Device device) const;
  OperatorBenchmark withDType(DType dtype) const;
  OperatorBenchmark withLoop(int numLoop) const;
  OperatorBenchmark withWarmUpLoop(int numLoop) const;

  void benchmarkAdd(lut::Span<const int> shape) const;
  void benchmarkSub(lut::Span<const int> shape) const;
  void benchmarkMul(lut::Span<const int> shape) const;

  void benchmarkAll();
  void printResult();

 private:
  enum OpType { OpAdd, OpSub, OpMul };

  Device _device;
  DType _dtype;
  int _numLoop;
  int _numWarmUpLoop;
  std::shared_ptr<std::vector<std::pair<std::string, double>>> _records;

  Tensor generateTensor(lut::Span<const int> shape) const;
  void addRecord(std::string name, double value) const;

  template<OpType OPTYPE>
  void benchmarkBinaryOperators(lut::Span<const int> shape, std::string_view name) const;
};

}  // namespace ly
