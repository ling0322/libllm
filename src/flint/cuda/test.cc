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

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/half.h"
#include "lutil/random.h"
#include "lutil/time.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/cuda/common.h"
#include "flint/cuda/cuda_operators.h"
#include "flint/cuda/dequant.h"
#include "flint/cuda/matmul.h"
#include "flint/cuda/matvec.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operator_tester.h"
#include "flint/operators.h"

using OperatorType = fl::OperatorTester::OperatorType;

namespace fl {

CATCH_TEST_CASE("test CUDA FastDivmod", "[fl][cuda]") {
  constexpr uint32_t divisors[] = {1, 2, 3, 7, 16, 255, 65535, INT32_MAX};

  for (uint32_t divisor : divisors) {
    op::cuda::FastDivmod divider(divisor);
    uint32_t dividends[] = {
        0, 1, divisor - 1, divisor, std::min(divisor + 1, uint32_t{INT32_MAX}), INT32_MAX};

    for (uint32_t dividend : dividends) {
      uint32_t quotient;
      uint32_t remainder;
      divider.divmod(dividend, quotient, remainder);
      CATCH_REQUIRE(quotient == dividend / divisor);
      CATCH_REQUIRE(remainder == dividend % divisor);
    }
  }
}

OperatorTester getOperatorTester() {
  return OperatorTester()
      .withOperators(getOperators(Device::kCuda))
      .withDevice(Device::getCuda())
      .withFloatType(DType::kFloat16);
}

CATCH_TEST_CASE("test CUDA lookup", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.testLookup());
}

CATCH_TEST_CASE("test CUDA repetitionPenalty", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.withTol(1e-3).testRepetitionPenalty());
}

CATCH_TEST_CASE("test CUDA matMul", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulSlice({10, 24}, {40, 64}));
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulSlice({5, 10, 24}, {40, 64}));
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulSlice({5, 10, 5, 24}, {10, 40, 64}));
}

CATCH_TEST_CASE("test CUDA binary operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.withTol(5e-3).testBinaryOp(OperatorType::Add));
  CATCH_REQUIRE(tester.withTol(5e-3).testBinaryOp(OperatorType::Sub));
  CATCH_REQUIRE(tester.testMulScale());
}

CATCH_TEST_CASE("test CUDA operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = OperatorTester()
                              .withOperators(getOperators(Device::kCuda))
                              .withDevice(Device::getCuda())
                              .withFloatType(DType::kFloat16);

  CATCH_SECTION("test basic operators") {
    CATCH_REQUIRE(tester.testToDevice({100, 200}));
    CATCH_REQUIRE(tester.testCast({100, 200}));

    CATCH_REQUIRE(tester.testCopy({10, 50}, true));
    CATCH_REQUIRE(tester.testCopy({2, 10, 50}, false));
    CATCH_REQUIRE(tester.testCopy({2, 10, 50}, true));
    CATCH_REQUIRE(tester.testCopy({2, 3, 10, 50}, true));
    CATCH_REQUIRE(tester.testCopyLongType());
    CATCH_REQUIRE(tester.testCopy5D());

    CATCH_REQUIRE(tester.testCausalMask());
  }

  CATCH_SECTION("test activations") {
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Softmax, {2, 5, 150}));
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Softmax, {2, 5, 151}));
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Swiglu, {2, 5, 150}));
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Swiglu, {2, 5, 152}));
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Softmax, {2, 5, 150}));
  }

  CATCH_SECTION("test normalizations") {
    CATCH_REQUIRE(tester.testRmsNorm({2, 5, 10}));
    CATCH_REQUIRE(tester.testRmsNorm({2, 5, 11}));
  }

  CATCH_SECTION("test positional embeddings") {
  }
}

CATCH_TEST_CASE("benchmark CUDA operators", "[op][cuda][benchmark]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = OperatorTester()
                              .withOperators(getOperators(Device::kCuda))
                              .withDevice(Device::getCuda())
                              .withFloatType(DType::kFloat16)
                              .withPrintBenchmarkInfo(true);

  CATCH_SECTION("benchmark copy") {
    CATCH_REQUIRE(tester.testCopy({2, 4096, 4096}, false));
    CATCH_REQUIRE(tester.testCopy({2, 4096, 4096}, true));
  }

  CATCH_SECTION("benchmark softmax") {
    CATCH_REQUIRE(tester.testUnaryOp(OperatorType::Softmax, {2, 256, 4096}));
    CATCH_REQUIRE(tester.testUnaryOp(OperatorType::Softmax, {2, 256, 50000}));
  }

  CATCH_SECTION("benchmark normalizations") {
    CATCH_REQUIRE(tester.testRmsNorm({2, 256, 4096}));
  }

  CATCH_SECTION("benchmark sampling") {
    Tensor distribution = F::rand({128256}, DType::kFloat);
    distribution = F::to(Device::getCuda(), distribution);
    distribution = F::cast(distribution, DType::kFloat16);
    for (int i = 0; i < 5; ++i) F::sample(distribution, 50, 1.0f);
    double start = lut::now();
    Tensor tokens;
    for (int i = 0; i < 20; ++i) tokens = F::sample(distribution, 50, 1.0f);
    tokens = F::to(Device::getCpu(), tokens);
    double milliseconds = (lut::now() - start) * 1000.0 / 20.0;
    LOG(INFO) << "GPU sample [128256]: " << milliseconds << "ms";
    CATCH_REQUIRE(tokens.getNumEl() == 1);
  }
}

CATCH_TEST_CASE("test softmax (large)", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<float>(
      {1, 1, 4},
      {
          -999.0f,
          -998.0f,
          -997.0f,
          -std::numeric_limits<float>::infinity(),
      });
  Tensor xr = F::softmax(a);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = F::softmax(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test softmax (strided)", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 3, 5}, DType::kFloat);
  Tensor xr = F::softmax(a.transpose(1, 2));

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = F::softmax(x.transpose(1, 2));
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 5e-3));
}

CATCH_TEST_CASE("test swiglu (strided)", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 3, 152}, DType::kFloat);
  Tensor xr = F::swiglu(a.transpose(0, 1));
  Tensor b = F::rand({2, 152, 3}, DType::kFloat);
  Tensor yr = F::swiglu(b.transpose(1, 2));

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = F::swiglu(x.transpose(0, 1));
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);
  Tensor y = F::to(Device::getCuda(), b);
  y = F::cast(y, DType::kFloat16);
  y = F::swiglu(y.transpose(1, 2));
  y = F::cast(y, DType::kFloat);
  y = F::to(Device::getCpu(), y);

  CATCH_REQUIRE(F::allClose(x, xr, 5e-3));
  CATCH_REQUIRE(F::allClose(y, yr, 5e-3));
}

CATCH_TEST_CASE("test rms norm (strided)", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 3, 11}, DType::kFloat);
  Tensor weight = F::rand({11}, DType::kFloat);
  Tensor xr = F::rmsNorm(a.transpose(0, 1), weight, 1e-5);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), weight);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  x = F::rmsNorm(x.transpose(0, 1), y, 1e-5);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 5e-3));
}

CATCH_TEST_CASE("test cat", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 10, 16}, DType::kFloat);
  Tensor b = F::rand({2, 2, 16}, DType::kFloat);
  Tensor xr = F::cat(a, b, 1);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  x = F::cat(x, y, 1);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test attention", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor q = F::rand({1, 16, 8, 16}, DType::kFloat);
  Tensor k = F::rand({1, 16, 8, 16}, DType::kFloat);
  Tensor v = F::rand({1, 16, 8, 16}, DType::kFloat);
  Tensor xr = F::attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), false);

  Tensor x = F::to(Device::getCuda(), q);
  Tensor y = F::to(Device::getCuda(), k);
  Tensor z = F::to(Device::getCuda(), v);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  z = F::cast(z, DType::kFloat16);
  x = x.transpose(1, 2);
  y = y.transpose(1, 2);
  z = z.transpose(1, 2);
  x = F::attention(x, y, z, false);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 5e-3f));
}

CATCH_TEST_CASE("test attention operator", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int numHeads = 8;
  constexpr int headDim = 128;

  auto runCase = [](int numKeyValueHeads, bool causal, int queryLength, int keyValueLength) {
    int groupSize = numHeads / numKeyValueHeads;
    Tensor q = F::rand({1, numHeads, queryLength, headDim}, DType::kFloat);
    Tensor k = F::rand({1, numKeyValueHeads, keyValueLength, headDim}, DType::kFloat);
    Tensor v = F::rand({1, numKeyValueHeads, keyValueLength, headDim}, DType::kFloat);

    auto expandHeads = [&](Tensor input) {
      Tensor expanded = input.unsqueeze(2).expand(
          {1, numKeyValueHeads, groupSize, keyValueLength, headDim});
      return F::contiguous(expanded).view({1, numHeads, keyValueLength, headDim});
    };

    Tensor scores = F::matmul(q, expandHeads(k).transpose(-2, -1));
    scores = F::mul(scores, 1.0f / sqrtf(1.0f * headDim));
    if (causal && queryLength > 1) {
      // The mask is aligned to the bottom right of the score matrix.
      Tensor mask =
          F::causalMask(keyValueLength).slice(0, {keyValueLength - queryLength, keyValueLength});
      scores = F::add(scores, mask);
    }
    Tensor xr = F::matmul(F::softmax(scores), expandHeads(v));

    Tensor x = F::cast(F::to(Device::getCuda(), q), DType::kFloat16);
    Tensor y = F::cast(F::to(Device::getCuda(), k), DType::kFloat16);
    Tensor z = F::cast(F::to(Device::getCuda(), v), DType::kFloat16);
    x = getOperators(Device::kCuda)->attention(x, y, z, causal);
    x = F::cast(x, DType::kFloat);
    x = F::to(Device::getCpu(), x);

    return F::allClose(x, xr, 5e-3f);
  };

  CATCH_CHECK(runCase(numHeads, false, 8, 8));
  CATCH_CHECK(runCase(numHeads, false, 128, 128));
  CATCH_CHECK(runCase(numHeads, true, 8, 8));
  CATCH_CHECK(runCase(2, false, 8, 8));
  CATCH_CHECK(runCase(2, true, 6, 10));
  CATCH_CHECK(runCase(2, true, 1, 10));

  // Long enough to make the operator split the keys.
  CATCH_CHECK(runCase(2, true, 1, 2048));
  CATCH_CHECK(runCase(2, true, 4, 1024));
}

CATCH_TEST_CASE("test CUDA sampling", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor probabilities = Tensor::create<float>(
      {3, 4},
      {0.4f, 0.3f, 0.2f, 0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.05f, 0.05f, 0.1f, 0.8f});
  Tensor probabilitiesFloat = F::to(Device::getCuda(), probabilities);
  Tensor probabilitiesHalf = F::cast(probabilitiesFloat, DType::kFloat16);

  Tensor sampled = F::sample(probabilitiesHalf, 1, 1.0f);
  CATCH_REQUIRE(sampled.getShape() == std::vector<int>{3});
  sampled = F::to(Device::getCpu(), sampled);
  const LongType *sampledData =
      sampled.getInternalData()->getData<LongType>(sampled.getInternalOffset());
  CATCH_REQUIRE(sampledData[0] == 0);
  CATCH_REQUIRE(sampledData[1] == 2);
  CATCH_REQUIRE(sampledData[2] == 3);

  F::manualSeed(Device::getCuda(), 1234);
  Tensor first = F::sample(probabilitiesFloat, 4, 0.9f);
  F::manualSeed(Device::getCuda(), 1234);
  Tensor second = F::sample(probabilitiesFloat, 4, 0.9f);
  first = F::to(Device::getCpu(), first);
  second = F::to(Device::getCpu(), second);
  const LongType *firstData = first.getInternalData()->getData<LongType>(first.getInternalOffset());
  const LongType *secondData =
      second.getInternalData()->getData<LongType>(second.getInternalOffset());
  for (int row = 0; row < 3; ++row) CATCH_REQUIRE(firstData[row] == secondData[row]);

  std::vector<float> values(2 * 2050, 0.0f);
  values[2049] = 0.4f;
  values[2050 + 1100] = 0.7f;
  Tensor multiBlock = Tensor::create<float>({2, 2050}, values);
  multiBlock = F::to(Device::getCuda(), multiBlock);
  multiBlock = F::cast(multiBlock, DType::kFloat16);
  Tensor multiBlockSampled = F::to(Device::getCpu(), F::sample(multiBlock, 1, 1.0f));
  const LongType *multiBlockData = multiBlockSampled.getInternalData()->getData<LongType>(
      multiBlockSampled.getInternalOffset());
  CATCH_REQUIRE(multiBlockData[0] == 2049);
  CATCH_REQUIRE(multiBlockData[1] == 1100);
}

CATCH_TEST_CASE("benchmark gemv", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor A = F::rand({8000, 4096}, DType::kFloat, Device::kCpu);
  Tensor x = F::rand({4096, 1}, DType::kFloat, Device::kCpu);

  A = F::to(Device::getCuda(), A);
  A = F::cast(A, DType::kFloat16);
  x = F::to(Device::getCuda(), x);
  x = F::cast(x, DType::kFloat16);

  LOG_TIME(F::matmul(A, x), "First call F::matmul(A, x)");
  LOG_TIME(Tensor x0 = F::matmul(A, x), "Second call F::matmul(A, x)");
  LOG_TIME(Tensor x1 = op::cuda::gemvHalf(A, x), "op::cuda::gemvHalf(A, x)");

  x0 = F::cast(x0, DType::kFloat);
  x1 = F::cast(x1, DType::kFloat);

  x0 = F::to(Device::getCpu(), x0);
  x1 = F::to(Device::getCpu(), x1);

  CATCH_REQUIRE(F::allClose(x0, x1));
}

#ifdef LIBLLM_CUTLASS_ENABLED

CATCH_TEST_CASE("test matmul gemm (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({10, 128}, DType::kFloat);
  Tensor b = F::rand({40, 256}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {128, 256}).transpose(1, 0));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {128, 256});
  y = y.transpose(1, 0);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-2f));
}

CATCH_TEST_CASE("test matmul bmm (cutlass)", "[fl][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({5, 10, 8, 24}, DType::kFloat);
  Tensor b = F::rand({10, 64, 24}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {8, 32}).transpose(-1, -2));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {8, 32});
  y = y.transpose(-1, -2);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 5e-3f));
}

CATCH_TEST_CASE("benchmark cutlass hgemm", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mmCutlass = op::cuda::MatMul::createCutlass();
  std::shared_ptr<op::cuda::MatMul> mmCublas = op::cuda::MatMul::createCublas();

  Tensor A = F::rand({512, 4096}, DType::kFloat);
  Tensor B = F::rand({4096, 4096}, DType::kFloat);
  A = F::to(Device::getCuda(), A);
  B = F::to(Device::getCuda(), B);
  A = F::cast(A, DType::kFloat16);
  B = F::cast(B, DType::kFloat16);

  Tensor Cr = mmCublas->apply(A, B);
  LOG_TIME(mmCublas->apply(A, B), "mmCublas->apply(A, B)");

  Tensor C = mmCutlass->apply(A, B);
  C = mmCutlass->apply(A, B);
  C = mmCutlass->apply(A, B);
  C = mmCutlass->apply(A, B);
  LOG_TIME(mmCutlass->apply(A, B), "mmCutlass->apply(A, B)");
}

Tensor toSm1xxScaleBlockRef(const Tensor &scale) {
  CHECK(scale.getDim() == 2);  //  && scale.getDType() == DType::kUInt8);

  int numRow = scale.getShape(0);
  int numCol = scale.getShape(1);

  CHECK(numRow % 128 == 0 && numCol % 4 == 0);
  Tensor x = F::contiguous(scale.view({numRow / 128, 128, numCol / 4, 4}).transpose(1, 2));
  x = F::contiguous(x.view({-1, 4, 32, 4}).transpose(1, 2));

  return x.view({-1, 32, 16});
}

#endif  // LIBLLM_CUTLASS_ENABLED

}  // namespace fl
