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
#include <limits>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/half.h"
#include "lutil/random.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/cuda/common.h"
#include "flint/cuda/cuda_operators.h"
#include "flint/cuda/dequant.h"
#include "flint/cuda/matmul.h"
#include "flint/cuda/matvec.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/memory.h"
#include "flint/operators.h"

namespace fl {

CATCH_TEST_CASE("test CUDA memory snapshot", "[fl][cuda][memory]") {
  MemorySnapshot::resetPeakStats(Device::getCuda());

  MemorySnapshot before = MemorySnapshot::capture(Device::getCuda());
  CATCH_REQUIRE(before.getTotalMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() <= before.getTotalMemory());

  int64_t bytes = 0;
  {
    Tensor x = F::tensor({1024, 1024}, DType::kFloat16, Device::getCuda());
    bytes = x.getNumEl() * 2;

    MemorySnapshot allocated = MemorySnapshot::capture(Device::getCuda());
    CATCH_REQUIRE(allocated.getAllocatedMemory() >= before.getAllocatedMemory() + bytes);
  }

  // the tensor is gone, but its bytes stay in the pool and remain in the peak.
  MemorySnapshot after = MemorySnapshot::capture(Device::getCuda());
  CATCH_REQUIRE(after.getAllocatedMemory() <= before.getAllocatedMemory());
  CATCH_REQUIRE(after.getPeakAllocatedMemory() >= bytes);
}

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

namespace {

// move to CUDA and cast to the type the CUDA backend computes in.
Tensor toCuda(const Tensor &a) {
  return F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
}

// bring a CUDA result back for comparison against the fp32 CPU reference.
Tensor toCpu(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

bool equalLong(Tensor a, Tensor b) {
  a.throwIfInvalidShape(b.getShape(), "equalLong");

  const LongType *pa = a.getInternalData()->getData<LongType>(a.getInternalOffset());
  const LongType *pb = b.getInternalData()->getData<LongType>(b.getInternalOffset());
  return std::equal(pa, pa + a.getNumEl(), pb);
}

}  // namespace

CATCH_TEST_CASE("test CUDA to and cast", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({100, 200}, DType::kFloat);

  Tensor roundTrip = F::to(Device::getCpu(), F::to(Device::getCuda(), a));
  CATCH_REQUIRE(F::allClose(roundTrip, a));

  CATCH_REQUIRE(F::allClose(toCpu(toCuda(a)), a));
}

CATCH_TEST_CASE("test CUDA copy", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](std::initializer_list<int> shape, bool transpose) {
    Tensor a = F::rand(shape, DType::kFloat);

    Tensor x = toCuda(a);
    if (transpose) x = x.transpose(1, 0);
    Tensor dest = F::tensorLike(x);
    F::copy(x, dest);

    dest = toCpu(dest);
    if (transpose) dest = dest.transpose(1, 0);
    return F::allClose(a, dest);
  };

  CATCH_REQUIRE(runCase({10, 50}, true));
  CATCH_REQUIRE(runCase({2, 10, 50}, false));
  CATCH_REQUIRE(runCase({2, 10, 50}, true));
  CATCH_REQUIRE(runCase({2, 3, 10, 50}, true));
}

CATCH_TEST_CASE("test CUDA copy (long)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<LongType>({2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});

  Tensor x = F::to(Device::getCuda(), a);
  Tensor dest = F::tensorLike(x);
  F::copy(x, dest);
  dest = F::to(Device::getCpu(), dest);

  CATCH_REQUIRE(equalLong(dest, a));
}

CATCH_TEST_CASE("test CUDA copy (expanded 5D)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({10, 2, 5, 20}, DType::kFloat);
  Tensor xr = F::contiguous(a.unsqueeze(1).expand({10, 4, 2, 5, 20}));

  Tensor x = toCuda(a).unsqueeze(1).expand({10, 4, 2, 5, 20});
  Tensor dest = F::tensorLike(x);
  F::copy(x, dest);

  CATCH_REQUIRE(F::allClose(toCpu(dest), xr));
}

CATCH_TEST_CASE("test CUDA lookup", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor embd = F::rand({10, 32}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor xr = F::lookup(embd, ids);

  Tensor x = F::lookup(toCuda(embd), F::to(Device::getCuda(), ids));

  CATCH_REQUIRE(F::allClose(toCpu(x), xr));

  // packed indices are 1D and give one embedding row per index.
  Tensor packedIds = Tensor::create<LongType>({3}, {1, 2, 3});
  Tensor packedRef = F::lookup(embd, packedIds);
  Tensor packed = F::lookup(toCuda(embd), F::to(Device::getCuda(), packedIds));

  CATCH_REQUIRE(packed.getShape() == std::vector<int>{3, 32});
  CATCH_REQUIRE(F::allClose(toCpu(packed), packedRef));
}

CATCH_TEST_CASE("test CUDA matmul", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](std::initializer_list<int> shapeA, std::initializer_list<int> shapeB) {
    Tensor a = F::rand(shapeA, DType::kFloat);
    Tensor b = F::rand(shapeB, DType::kFloat);
    Tensor xr = F::matmul(a, b.slice(-1, {8, 32}).transpose(-1, -2));

    Tensor y = toCuda(b).slice(-1, {8, 32}).transpose(-1, -2);
    Tensor x = F::matmul(toCuda(a), y);

    return F::allClose(toCpu(x), xr, 5e-2);
  };

  CATCH_REQUIRE(runCase({10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 5, 24}, {10, 40, 64}));
}

CATCH_TEST_CASE("test CUDA binary operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({5}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toCuda(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toCuda(b);

  CATCH_REQUIRE(F::allClose(toCpu(F::add(xt, y)), F::add(at, b), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sub(xt, y)), F::sub(at, b), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::mul(xt, 0.1f)), F::mul(at, 0.1f), 1e-3, 1e-4));
}

CATCH_TEST_CASE("test CUDA scalar operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::div(toCuda(a), 8.0f)), F::mul(a, 1.0f / 8.0f), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::square(toCuda(a))), F::mul(a, a), 5e-3));

  Tensor ids = Tensor::create<LongType>({2, 3}, {0, 1, 2, 3, 4, 5});
  Tensor mod = F::to(Device::getCpu(), F::mod(F::to(Device::getCuda(), ids), 3));
  CATCH_REQUIRE(equalLong(mod, Tensor::create<LongType>({2, 3}, {0, 1, 2, 0, 1, 2})));

  CATCH_REQUIRE(F::elem(toCuda(Tensor::create<float>({1}, {1.5f}))) == 1.5f);
}

CATCH_TEST_CASE("test CUDA activations", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {150, 151}) {
    Tensor a = F::rand({2, 5, lastDim}, DType::kFloat);
    CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toCuda(a))), F::softmax(a), 5e-3));
  }

  for (int lastDim : {150, 152}) {
    Tensor a = F::rand({2, 5, lastDim}, DType::kFloat);
    CATCH_REQUIRE(F::allClose(toCpu(F::swiglu(toCuda(a))), F::swiglu(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA activations (strided)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 3, 5}, DType::kFloat);
  Tensor x = F::softmax(toCuda(a).transpose(1, 2));
  CATCH_REQUIRE(F::allClose(toCpu(x), F::softmax(a.transpose(1, 2)), 5e-3));

  Tensor b = F::rand({2, 3, 152}, DType::kFloat);
  Tensor y = F::swiglu(toCuda(b).transpose(0, 1));
  CATCH_REQUIRE(F::allClose(toCpu(y), F::swiglu(b.transpose(0, 1)), 5e-3));

  Tensor c = F::rand({2, 152, 3}, DType::kFloat);
  Tensor z = F::swiglu(toCuda(c).transpose(1, 2));
  CATCH_REQUIRE(F::allClose(toCpu(z), F::swiglu(c.transpose(1, 2)), 5e-3));
}

CATCH_TEST_CASE("test CUDA softmax (extreme values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<float>(
      {1, 1, 4},
      {-999.0f, -998.0f, -997.0f, -std::numeric_limits<float>::infinity()});

  CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toCuda(a))), F::softmax(a)));
}

CATCH_TEST_CASE("test CUDA rmsNorm", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {10, 11}) {
    Tensor a = F::rand({2, 5, lastDim}, DType::kFloat);
    Tensor w = F::rand({lastDim}, DType::kFloat);
    Tensor x = F::rmsNorm(toCuda(a), toCuda(w), 1e-5);
    CATCH_REQUIRE(F::allClose(toCpu(x), F::rmsNorm(a, w, 1e-5), 5e-3));
  }

  // strided input.
  Tensor a = F::rand({2, 3, 11}, DType::kFloat);
  Tensor w = F::rand({11}, DType::kFloat);
  Tensor x = F::rmsNorm(toCuda(a).transpose(0, 1), toCuda(w), 1e-5);
  CATCH_REQUIRE(F::allClose(toCpu(x), F::rmsNorm(a.transpose(0, 1), w, 1e-5), 5e-3));
}

CATCH_TEST_CASE("test CUDA reductions", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 5, 150}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::sum(toCuda(a))), F::sum(a), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::max(toCuda(a))), F::max(a), 5e-3));
}

CATCH_TEST_CASE("test CUDA tensor creation", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor filled = F::tensor({2, 5, 10}, DType::kFloat16, Device::getCuda());
  F::fill(filled, 1.5f);
  Tensor filledRef = F::tensor({2, 5, 10}, DType::kFloat, Device::getCpu());
  F::fill(filledRef, 1.5f);
  CATCH_REQUIRE(F::allClose(toCpu(filled), filledRef));

  Tensor zeros = F::zeros({2, 5, 10}, DType::kFloat16, Device::getCuda());
  CATCH_REQUIRE(F::allClose(toCpu(zeros), F::zeros({2, 5, 10}, DType::kFloat)));

  Tensor arange = F::to(Device::getCpu(), F::arange(0, 10, 2, Device::getCuda()));
  CATCH_REQUIRE(equalLong(arange, Tensor::create<LongType>({5}, {0, 2, 4, 6, 8})));
}

CATCH_TEST_CASE("test CUDA randNormal", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor x = toCpu(getOperators(Device::kCuda)->randNormal({4096}));

  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  double sum = 0.0;
  double sumSquare = 0.0;
  for (int i = 0; i < x.getNumEl(); ++i) {
    sum += data[i];
    sumSquare += data[i] * data[i];
  }

  double mean = sum / x.getNumEl();
  double stddev = sqrt(sumSquare / x.getNumEl() - mean * mean);
  CATCH_REQUIRE(fabs(mean) < 0.1);
  CATCH_REQUIRE(fabs(stddev - 1.0) < 0.1);
}

CATCH_TEST_CASE("test CUDA causalMask", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int Dim = 129;
  Tensor xr = F::softmax(F::causalMask(Dim));
  Tensor x = F::softmax(F::causalMask(Dim, Device::getCuda()));

  CATCH_REQUIRE(F::allClose(toCpu(x), xr, 1e-3, 1e-4));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 16}, DType::kFloat);
  Tensor history = Tensor::create<LongType>({2, 4}, {1, 0, 1, 3, 0, 0, 0, 1});

  Tensor x = toCuda(a);
  F::repetitionPenalty(x, F::to(Device::getCuda(), history), 1.5);
  F::repetitionPenalty(a, history, 1.5);

  CATCH_REQUIRE(F::allClose(toCpu(x), a, 1e-3));
}

CATCH_TEST_CASE("test CUDA attention", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](int numHeads,
                    int numKeyValueHeads,
                    int queryLength,
                    int keyValueLength,
                    int headDim,
                    bool causal) {
    // a model feeds attention with [batch, length, heads, headDim] transposed to
    // [batch, heads, length, headDim], so the inputs are not contiguous.
    Tensor q = F::rand({1, queryLength, numHeads, headDim}, DType::kFloat);
    Tensor k = F::rand({1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
    Tensor v = F::rand({1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
    Tensor xr = F::attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal);

    Tensor x = F::attention(
        toCuda(q).transpose(1, 2),
        toCuda(k).transpose(1, 2),
        toCuda(v).transpose(1, 2),
        causal);

    return F::allClose(toCpu(x), xr, 5e-3f);
  };

  // headDim 128 goes to FlashAttention.
  CATCH_REQUIRE(runCase(8, 8, 8, 8, 128, false));
  CATCH_REQUIRE(runCase(8, 8, 128, 128, 128, false));
  CATCH_REQUIRE(runCase(8, 8, 8, 8, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 8, 8, 128, false));
  CATCH_REQUIRE(runCase(8, 2, 6, 10, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 1, 10, 128, true));

  // long enough to make the operator split the keys.
  CATCH_REQUIRE(runCase(8, 2, 1, 2048, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 4, 1024, 128, true));

  // headDim 16 is unsupported by FlashAttention and falls back to the portable path.
  CATCH_REQUIRE(runCase(8, 8, 16, 16, 16, false));
  CATCH_REQUIRE(runCase(8, 2, 6, 10, 16, true));
}

CATCH_TEST_CASE("test CUDA cat", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 10, 16}, DType::kFloat);
  Tensor b = F::rand({2, 2, 16}, DType::kFloat);

  Tensor x = F::cat(toCuda(a), toCuda(b), 1);

  CATCH_REQUIRE(F::allClose(toCpu(x), F::cat(a, b, 1), 5e-3));
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

CATCH_TEST_CASE("test gemv", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor w = F::rand({8000, 4096}, DType::kFloat, Device::kCpu);
  Tensor x = F::rand({1, 4096}, DType::kFloat, Device::kCpu);

  w = F::cast(F::to(Device::getCuda(), w), DType::kFloat16);
  x = F::cast(F::to(Device::getCuda(), x), DType::kFloat16);

  // the layout Linear::forward feeds to matmul: a transposed weight, so getStride(0) == 1.
  Tensor wT = w.transpose(0, 1);

  Tensor xr = F::matmul(x, F::contiguous(wT));
  Tensor xv = op::cuda::gemvHalf(x.subtensor(0), wT);

  xr = F::to(Device::getCpu(), F::cast(xr, DType::kFloat));
  xv = F::to(Device::getCpu(), F::cast(xv, DType::kFloat));

  CATCH_REQUIRE(F::allClose(xr, xv, 5e-3f));
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
