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

#include "lynn/operator_benchmark.h"

#include "lutil/strings.h"

namespace ly {

// 构造函数
OperatorBenchmark::OperatorBenchmark()
    : _device(Device::getCpu()),
      _numLoop(10),
      _dtype(DType::kFloat),
      _numWarmUpLoop(5),
      _records(std::make_shared<std::vector<std::pair<std::string, double>>>()) {
}

OperatorBenchmark OperatorBenchmark::withOperators(Device device) const {
  OperatorBenchmark b = *this;
  b._device = device;
  return b;
}

OperatorBenchmark OperatorBenchmark::withDType(DType dtype) const {
  OperatorBenchmark b = *this;
  b._dtype = dtype;
  return b;
}

OperatorBenchmark OperatorBenchmark::withLoop(int numLoop) const {
  OperatorBenchmark b = *this;
  b._numLoop = numLoop;
  return b;
}

OperatorBenchmark OperatorBenchmark::withWarmUpLoop(int warmUpLoop) const {
  OperatorBenchmark b = *this;
  b._numWarmUpLoop = warmUpLoop;
  return b;
}

template<OperatorBenchmark::OpType OPTYPE>
void OperatorBenchmark::benchmarkBinaryOperators(lut::Span<const int> shape, std::string_view name)
    const {
  Tensor a = generateTensor(shape);
  Tensor b = generateTensor(shape);

  double t0;
  for (int i = 0; i < _numLoop + _numWarmUpLoop; ++i) {
    if (i == _numWarmUpLoop) t0 = lut::now();

    if constexpr (OPTYPE == OpAdd) {
      Tensor c = a + b;
    } else if constexpr (OPTYPE == OpSub) {
      Tensor c = a - b;
    } else if constexpr (OPTYPE == OpMul) {
      Tensor c = a * b;
    } else {
      NOT_IMPL();
    }
  }

  double t1 = lut::now();
  std::string record = lut::sprintf(
      "%s:%s,%s",
      std::string(name),
      a.getInternalShape()->toString(),
      _dtype.toString());
  addRecord(record, (t1 - t0) / _numLoop);
}

void OperatorBenchmark::printResult() {
  for (const std::pair<std::string, double> &r : *_records) {
    printf("%s", r.first.c_str());
    for (int i = 0; i < 32 - int(r.first.size()); ++i) {
      putchar(' ');
    }
    printf("%.6lf\n", r.second);
  }
}

void OperatorBenchmark::benchmarkAdd(lut::Span<const int> shape) const {
  benchmarkBinaryOperators<OpAdd>(shape, "add");
}

void OperatorBenchmark::benchmarkSub(lut::Span<const int> shape) const {
  benchmarkBinaryOperators<OpSub>(shape, "sub");
}

void OperatorBenchmark::benchmarkMul(lut::Span<const int> shape) const {
  benchmarkBinaryOperators<OpMul>(shape, "mul");
}

Tensor OperatorBenchmark::generateTensor(lut::Span<const int> shape) const {
  return F::rand(shape, _dtype, _device);
}

void OperatorBenchmark::addRecord(std::string name, double value) const {
  name = lut::replace(name, " ", "");
  _records->emplace_back(name, value);
}

void OperatorBenchmark::benchmarkAll() {
  benchmarkAdd({4096, 4096});
  benchmarkSub({4096, 4096});
  benchmarkMul({4096, 4096});
}

}  // namespace ly
