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
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/cpu/cpu_tensor_data.h"
#include "libllm/cuda/dequant.h"
#include "libllm/cuda/matmul.h"
#include "libllm/cuda/matvec.h"
#include "libllm/device.h"
#include "libllm/functional.h"
#include "libllm/operator_tester.h"
#include "lut/half.h"
#include "lut/random.h"
#include "lut/time.h"

#define CONCAT2(l, r) l##r
#define CONCAT(l, r) CONCAT2(l, r)

#define LOG_TIME(stmt, message)             \
  double CONCAT(t0, __LINE__) = lut::now(); \
  stmt;                                     \
  LOG(INFO) << message << ": " << (lut::now() - CONCAT(t0, __LINE__)) * 1000 << "ms";

using OperatorType = libllm::OperatorTester::OperatorType;

namespace libllm {

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
  CATCH_REQUIRE(tester.testLookupQInt4());
}

CATCH_TEST_CASE("test CUDA unfold", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.withTol(2e-3).testUnfold());
}

CATCH_TEST_CASE("test CUDA layerNorm", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.withTol(5e-3).testLayerNorm({1, 1, 160}));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.withTol(1e-3).testRepetitionPenalty());
}

CATCH_TEST_CASE("test CUDA matMul", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulQInt4({1, 1, 128}, {50, 128}, true));
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulQInt4({5, 10, 50}, {50, 128}, false));
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulSlice({10, 20}, {40, 30}));
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulSlice({5, 10, 20}, {40, 30}));
  CATCH_REQUIRE(tester.withTol(5e-2).testMatmulSlice({5, 10, 5, 20}, {10, 40, 30}));
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

    CATCH_REQUIRE(tester.testCopy({2, 10, 50}, false));
    CATCH_REQUIRE(tester.testCopy({2, 10, 50}, true));
    CATCH_REQUIRE(tester.testCopyLongType());
    CATCH_REQUIRE(tester.testCopy5D());

    CATCH_REQUIRE(tester.testCausalMask());
  }

  CATCH_SECTION("test binary operators") {
    CATCH_REQUIRE(tester.testMulScale());
    // CATCH_REQUIRE(tester.testBinaryOp(OperatorType::Mul));
    CATCH_REQUIRE(tester.withTol(5e-3).testBinaryOp(OperatorType::Add));
  }

  CATCH_SECTION("test activations") {
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Softmax, {2, 5, 150}));
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Swiglu, {2, 5, 150}));
    CATCH_REQUIRE(tester.withTol(5e-3).testUnaryOp(OperatorType::Gelu, {2, 5, 150}));
  }

  CATCH_SECTION("test normalizations") {
    CATCH_REQUIRE(tester.testRmsNorm({2, 5, 10}));
  }

  CATCH_SECTION("test positional embeddings") {
    CATCH_REQUIRE(tester.testRoPE());
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
    CATCH_REQUIRE(tester.testLayerNorm({2, 256, 4096}));
  }
}

CATCH_TEST_CASE("test dequant", "[ly][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({5, 256}, DType::kQInt4x32);
  Tensor xr = F::cast(a, DType::kFloat);

  Tensor x = F::to(Device::getCuda(), a);
  x = op::cuda::dequantQ4ToHalf(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test softmax (large)", "[ly][op][cuda]") {
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

CATCH_TEST_CASE("test cat", "[ly][op][cuda]") {
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

CATCH_TEST_CASE("test attention", "[ly][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor q = F::rand({1, 2, 5, 16}, DType::kFloat);
  Tensor k = F::rand({1, 2, 5, 16}, DType::kFloat);
  Tensor v = F::rand({1, 2, 5, 16}, DType::kFloat);
  Tensor xr = F::attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2));

  Tensor x = F::to(Device::getCuda(), q);
  Tensor y = F::to(Device::getCuda(), k);
  Tensor z = F::to(Device::getCuda(), v);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  z = F::cast(z, DType::kFloat16);
  x = x.transpose(1, 2);
  y = y.transpose(1, 2);
  z = z.transpose(1, 2);
  x = F::attention(x, y, z);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 2e-3f));
}

CATCH_TEST_CASE("benchmark gemv", "[ly][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  lut::Random r(0x55aa);
  Tensor Aq = F::rand({8000, 4096}, DType::kQInt4x32, Device::kCpu, &r);
  Tensor x = F::rand({4096, 1}, DType::kFloat, Device::kCpu, &r);
  Tensor xrq = F::matmul(x.transpose(0, 1), Aq.transpose(0, 1));

  Aq = F::to(Device::getCuda(), Aq);
  x = F::to(Device::getCuda(), x);
  x = F::cast(x, DType::kFloat16);

  Tensor A = op::cuda::dequantQ4ToHalf(Aq);

  LOG_TIME(F::matmul(A, x), "First call F::matmul(A, x)");
  LOG_TIME(Tensor x0 = F::matmul(A, x), "Second call F::matmul(A, x)");
  LOG_TIME(Tensor x1 = op::cuda::gemvHalf(A, x), "op::cuda::gemvHalf(A, x)");
  LOG_TIME(Tensor xq = op::cuda::gemvQ4(Aq, x), "op::cuda::gemvQ4(Aq, x)");

  x0 = F::cast(x0, DType::kFloat);
  x1 = F::cast(x1, DType::kFloat);
  xq = F::cast(xq, DType::kFloat);

  x0 = F::to(Device::getCpu(), x0);
  x1 = F::to(Device::getCpu(), x1);
  xq = F::to(Device::getCpu(), xq);

  CATCH_REQUIRE(F::allClose(x0, xrq.transpose(0, 1), 5e-3f));
  CATCH_REQUIRE(F::allClose(x1, xrq.transpose(0, 1), 5e-3f));
  CATCH_REQUIRE(F::allClose(xq, xrq.transpose(0, 1), 5e-3f));
}

#ifdef LIBLLM_CUTLASS_ENABLED

CATCH_TEST_CASE("test matmul gemm (cutlass)", "[ly][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({10, 20}, DType::kFloat);
  Tensor b = F::rand({40, 30}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(1, 0));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(1, 0);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f));
}

CATCH_TEST_CASE("test matmul bmm (cutlass)", "[ly][op][cuda][cutlass]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mm = op::cuda::MatMul::createCutlass();

  Tensor a = F::rand({5, 10, 5, 20}, DType::kFloat);
  Tensor b = F::rand({10, 30, 20}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(-1, -2));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(-1, -2);
  x = mm->apply(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f));
}

CATCH_TEST_CASE("benchmark cutlass hgemm", "[ly][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::shared_ptr<op::cuda::MatMul> mmCutlass = op::cuda::MatMul::createCutlass();
  std::shared_ptr<op::cuda::MatMul> mmCublas = op::cuda::MatMul::createCublas();

  Tensor A = F::rand({4096, 4096}, DType::kFloat);
  Tensor B = F::rand({4096, 4096}, DType::kFloat);
  A = F::to(Device::getCuda(), A);
  B = F::to(Device::getCuda(), B);
  A = F::cast(A, DType::kFloat16);
  B = F::cast(B, DType::kFloat16);

  Tensor Cr = mmCublas->apply(A, B);
  LOG_TIME(mmCublas->apply(A, B), "mmCublas->apply(A, B)");

  Tensor C = mmCublas->apply(A, B);
  LOG_TIME(mmCutlass->apply(A, B), "mmCutlass->apply(A, B)");
}

#endif  // LIBLLM_CUTLASS_ENABLED

}  // namespace libllm
