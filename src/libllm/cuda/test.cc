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

#include "catch2/catch_amalgamated.hpp"

#include <cuda_fp16.h>
#include <algorithm>
#include "libllm/device.h"
#include "libllm/functional.h"
#include "libllm/cpu/cpu_tensor_data.h"
#include "libllm/cuda/dequant.h"
#include "libllm/cuda/matmul.h"
#include "libllm/cuda/matvec.h"
#include "libllm/lut/half.h"
#include "libllm/lut/random.h"
#include "libllm/lut/time.h"

#define CONCAT2(l, r) l ## r
#define CONCAT(l, r) CONCAT2(l, r)

#define LOG_TIME(stmt, message) \
  double CONCAT(t0, __LINE__) = lut::now(); \
  stmt; \
  LOG(INFO) << message <<  ": " << (lut::now() - CONCAT(t0, __LINE__)) * 1000 << "ms";


namespace libllm {

/// @brief Create a random 2D tensor with Q4 type.
Tensor createRandomQ4Tensor2D(int numRow, int numCol) {
  constexpr int groupSize = Q4::GroupSize;
  CHECK(numCol % groupSize == 0);
  int numel = numRow * numCol;

  std::vector<uint8_t> data(numel / 2);
  std::vector<float> scaleFloat(numel / groupSize);
  std::vector<uint16_t> scale(numel / groupSize);
  std::vector<uint8_t> zero(numel / groupSize / 2);

  int magicNumber = 0x322;
  lut::Random random(magicNumber);
  random.fillUInt8(lut::makeSpan(data));
  random.fillUInt8(lut::makeSpan(zero));
  random.fill(lut::makeSpan(scaleFloat), 0, 0.1);
  std::transform(scaleFloat.begin(), scaleFloat.end(), scale.begin(), lut::cvtss_sh);

  auto tensorData = op::cpu::CpuTensorData::create({
      {numel, DType::kQ4},
      {numel / groupSize, DType::kFloat16},
      {numel / groupSize / 2, DType::kUInt8}});
  memcpy(tensorData->getSlot(0)->getRawData(), data.data(), data.size());
  memcpy(tensorData->getSlot(1)->getRawData(), scale.data(), scale.size() * sizeof(uint16_t));
  memcpy(tensorData->getSlot(2)->getRawData(), zero.data(), zero.size());

  std::initializer_list<int> shape{numRow, numCol};
  auto tensorShape = std::make_shared<TensorShape>(shape);

  return Tensor::create(tensorShape, tensorData);
}

CATCH_TEST_CASE("test cuda toDevice", "[ly][op][cuda]") {
  Tensor xr = F::rand({100, 200}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), xr);
  x = F::to(Device::getCpu(), x);
  
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test cuda cast", "[ly][op][cuda]") {
  Tensor xr = F::rand({100, 20, 50}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), xr);
  x = F::cast(x, DType::kFloat16);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test cuda copy", "[ly][op][cuda]") {
  Tensor tensor = F::rand({2, 5, 20}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), tensor);

  // cudaMemcpy path.
  x = F::cast(x, DType::kFloat16);
  Tensor x2 = F::createTensorLike(x);
  F::copy(x, x2);
  x2 = F::cast(x2, DType::kFloat);
  x2 = F::to(Device::getCpu(), x2);
  CATCH_REQUIRE(F::allClose(tensor, x2));

  // other path.
  x = F::to(Device::getCuda(), tensor);
  x = F::cast(x, DType::kFloat16);
  x = x.transpose(1, 0);
  x2 = F::createTensorLike(x);
  F::copy(x, x2);

  x2 = F::cast(x2, DType::kFloat);
  x2 = F::to(Device::getCpu(), x2);

  CATCH_REQUIRE(F::allClose(tensor, x2.transpose(1, 0)));
}

CATCH_TEST_CASE("benchmark copy", "[ly][op][cuda]") {
  Tensor tensor = F::rand({2, 4096, 4096}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), tensor);

  // other path.
  x = F::to(Device::getCuda(), tensor);
  x = F::cast(x, DType::kFloat16);
  x = x.transpose(1, 0);
  Tensor x2 = F::createTensorLike(x);

  LOG_TIME(F::copy(x, x2), "F::copy(x, x2)");

  x2 = F::cast(x2, DType::kFloat);
  x2 = F::to(Device::getCpu(), x2);

  CATCH_REQUIRE(F::allClose(tensor, x2.transpose(1, 0)));
}

CATCH_TEST_CASE("test cuda copy (int64_t)", "[ly][op][cuda]") {
  Tensor tensor = Tensor::create<LongType>({2, 5}, {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 0
  });

  // cudaMemcpy path (op::cuda::transform path is not supported for int16_t)
  Tensor x = F::to(Device::getCuda(), tensor);
  Tensor x2 = F::createTensorLike(x);
  F::copy(x, x2);
  x2 = F::to(Device::getCpu(), x2);

  const LongType *px = tensor.getData<LongType>(), 
                       *pr = x2.getData<LongType>();
  x2.throwIfInvalidShape(tensor.getShape());
  CATCH_REQUIRE(std::equal(px, px + x2.getNumEl(), pr));
}

CATCH_TEST_CASE("test cuda contiguous (copy) 5D", "[ly][op][cuda]") {
  Tensor tensor = F::rand({10, 2, 5, 20}, DType::kFloat);

  Tensor x = F::to(Device::getCuda(), tensor);
  x = F::cast(x, DType::kFloat16);
  x = x.unsqueeze(1).expand({10, 4, 2, 5, 20});
  x = F::contiguous(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  Tensor xr = tensor.unsqueeze(1).expand({10, 4, 2, 5, 20});
  xr = F::contiguous(x);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test lookup half", "[ly][op][cuda]") {
  Tensor embd = F::rand({10, 20}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = F::to(Device::getCuda(), embd);
  x = F::cast(x, DType::kFloat16);

  Tensor y = F::to(Device::getCuda(), ids);
  x = F::lookup(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  Tensor xr = F::lookup(embd, ids);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test lookup q4", "[ly][op][cuda]") {
  Tensor embd = createRandomQ4Tensor2D(10, 256);
  Tensor ids = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = F::to(Device::getCuda(), embd);
  Tensor y = F::to(Device::getCuda(), ids);
  x = F::lookup(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  Tensor xr = F::lookup(embd, ids);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test matmul gemm", "[ly][op][cuda]") {
  Tensor a = F::rand({10, 20}, DType::kFloat);
  Tensor b = F::rand({40, 30}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(1, 0));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(1, 0);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test matmul bmm (gemm)", "[ly][op][cuda]") {
  Tensor a = F::rand({5, 10, 20}, DType::kFloat);
  Tensor b = F::rand({40, 30}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(1, 0));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(1, 0);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test matmul bmm", "[ly][op][cuda]") {
  Tensor a = F::rand({5, 10, 5, 20}, DType::kFloat);
  Tensor b = F::rand({10, 30, 20}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(-1, -2));

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(-1, -2);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test matmul q4", "[ly][op][cuda]") {
  Tensor a = F::rand({5, 10, 50}, DType::kFloat);
  Tensor b = createRandomQ4Tensor2D(50, 128);
  Tensor xr = F::matmul(a, b);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test q4 matmul (gemv)", "[ly][op][cuda]") {
  Tensor Aq = createRandomQ4Tensor2D(8000, 4096);
  Tensor x = F::rand({4096, 1}, DType::kFloat);
  Tensor xr = F::matmul(x.transpose(0, 1), Aq.transpose(0, 1));

  Aq = F::to(Device::getCuda(), Aq);
  x = F::to(Device::getCuda(), x);
  x = F::cast(x, DType::kFloat16);
  x = F::matmul(x.transpose(0, 1), Aq.transpose(0, 1));
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 5e-2f));
}

CATCH_TEST_CASE("test scale", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor xr = F::mul(a.transpose(2, 1).slice(1, {1, 9}), 0.1f);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = x.transpose(2, 1);
  x = x.slice(1, {1, 9});
  x = F::mul(x, 0.1f);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test mul", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({5}, DType::kFloat);
  Tensor xr = F::mul(a.transpose(2, 1).slice(1, {1, 9}), b);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  x = x.transpose(2, 1);
  x = x.slice(1, {1, 9});
  x = F::mul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test softmax", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 5, 150}, DType::kFloat);
  Tensor xr = F::softmax(a);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = F::softmax(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("benchmark softmax", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 256, 4096}, DType::kFloat);
  Tensor xr = F::softmax(a);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  LOG_TIME(x = F::softmax(x), "d=4096 F::softmax(x)");
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("benchmark softmax large", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 256, 50000}, DType::kFloat);
  Tensor xr = F::softmax(a);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  LOG_TIME(x = F::softmax(x), "d=50000 F::softmax(x)");
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test add", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({10}, DType::kFloat);
  Tensor xr = F::add(a, b);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  x = F::add(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test rms_norm", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({10}, DType::kFloat);
  Tensor xr = F::rmsNorm(a, b, 1e-5);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  x = F::rmsNorm(x, y, 1e-5);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 1e-2f));
}

CATCH_TEST_CASE("benchmark rms_norm", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 256, 4096}, DType::kFloat);
  Tensor b = F::rand({4096}, DType::kFloat);
  Tensor xr = F::rmsNorm(a, b, 1e-5);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  LOG_TIME(x = F::rmsNorm(x, y, 1e-5), "d=4096 F::rmsNorm(x, y, 1e-5)");
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 1e-2f));
}

CATCH_TEST_CASE("test causal_mask", "[ly][op][cuda]") {
  constexpr int DIM = 129;
  Tensor xr = F::softmax(F::causalMask(DIM));
  Tensor x = F::softmax(F::causalMask(DIM, Device::getCuda()));
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}


CATCH_TEST_CASE("test apply_rope", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 5, 2, 16}, DType::kFloat);
  Tensor b = F::rand({5, 1, 16}, DType::kFloat);
  Tensor xr = F::applyRotaryPosEmb(a, b);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  x = F::applyRotaryPosEmb(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-3, 1e-3));
}

CATCH_TEST_CASE("test swiglu", "[ly][op][cuda]") {
  Tensor a = F::rand({2, 10, 16}, DType::kFloat);
  Tensor xr = F::swiglu(a);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = F::swiglu(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test dequant", "[ly][op][cuda]") {
  Tensor a = createRandomQ4Tensor2D(5, 256);
  Tensor xr = F::cast(a, DType::kFloat);

  Tensor x = F::to(Device::getCuda(), a);
  x = op::cuda::dequantQ4ToHalf(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test cat", "[ly][op][cuda]") {
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

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("benchmark gemv", "[ly][op][cuda]") {
  Tensor Aq = createRandomQ4Tensor2D(8000, 4096);
  Tensor x = F::rand({4096, 1}, DType::kFloat);
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

  CATCH_REQUIRE(F::allClose(x0, xrq.transpose(0, 1), 1e-5f, 5e-2f));
  CATCH_REQUIRE(F::allClose(x1, xrq.transpose(0, 1), 1e-5f, 5e-2f));
  CATCH_REQUIRE(F::allClose(xq, xrq.transpose(0, 1), 1e-5f, 5e-2f));
}

#ifdef LIBLLM_CUTLASS_ENABLED

CATCH_TEST_CASE("test matmul gemm (cutlass)", "[ly][op][cuda][cutlass]") {
  std::shared_ptr<MatMul> mm = MatMul::createCutlass();

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

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test matmul bmm (cutlass)", "[ly][op][cuda][cutlass]") {
  std::shared_ptr<MatMul> mm = MatMul::createCutlass();

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

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("benchmark cutlass hgemm", "[ly][op][cuda]") {
  std::shared_ptr<MatMul> mmCutlass = MatMul::createCutlass();
  std::shared_ptr<MatMul> mmCublas = MatMul::createCublas();

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
