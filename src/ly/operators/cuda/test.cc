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

#include "../../../../third_party/catch2/catch_amalgamated.hpp"

#include <cuda_fp16.h>
#include <algorithm>
#include "ly/device.h"
#include "ly/ly.h"
#include "ly/operators/cpu/cpu_tensor_data.h"
#include "ly/operators/cuda/matvec.h"
#include "ly/operators/cuda/dequant.h"
#include "ly/internal/tensor_shape.h"
#include "lyutil/half.h"
#include "lyutil/random.h"
#include "lyutil/time.h"

using ly::Tensor;
using ly::DType;
using ly::Device;
using ly::internal::TensorShape;
using ly::op::cpu::CpuTensorData;

namespace F = ly::functional;

/// @brief Create a random 2D tensor with Q4 type.
Tensor createRandomQ4Tensor2D(int numRow, int numCol) {
  constexpr int groupSize = ly::QInt4Group32::GroupSize;
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

  auto tensorData = CpuTensorData::create({
      {numel, DType::kQInt4Group32},
      {numel / groupSize, DType::kFloat16},
      {numel / groupSize / 2, DType::kUInt8}});
  memcpy(tensorData->getSlot(0)->getRawData(), data.data(), data.size());
  memcpy(tensorData->getSlot(1)->getRawData(), scale.data(), scale.size() * sizeof(uint16_t));
  memcpy(tensorData->getSlot(2)->getRawData(), zero.data(), zero.size());

  std::initializer_list<int> shape{numRow, numCol};
  auto tensorShape = std::make_shared<TensorShape>(shape);

  return Tensor::create(tensorShape, tensorData);
}

CATCH_TEST_CASE("test cuda toDevice", "[cuda][operators][toDevice]") {
  Tensor xr = F::rand({100, 200}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), xr, false);
  x = F::to(Device::getCpu(), x, false);
  
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test cuda cast", "[cuda][operators][cast]") {
  Tensor xr = F::rand({100, 20, 50}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), xr, false);
  x = F::cast(x, DType::kFloat16);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test cuda copy", "[cuda][operators][copy]") {
  Tensor tensor = F::rand({100, 20}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), tensor);

  // cudaMemcpy path.
  x = F::cast(x, DType::kFloat16);
  Tensor x2 = F::createTensorLike(x);
  F::copy(x, x2);
  x2 = F::cast(x2, DType::kFloat);
  x2 = F::to(Device::getCpu(), x2);
  CATCH_REQUIRE(F::allClose(tensor, x2));

  // cudnnTransformTensor path.
  x = F::to(Device::getCuda(), tensor);
  x = F::cast(x, DType::kFloat16);
  x = x.transpose(1, 0);
  x2 = F::createTensorLike(x);
  F::copy(x, x2);

  x2 = F::cast(x2, DType::kFloat);
  x2 = F::to(Device::getCpu(), x2);
  CATCH_REQUIRE(F::allClose(tensor, x2.transpose(1, 0)));
}

CATCH_TEST_CASE("test cuda copy (int64_t)", "[cuda][operators][copy]") {
  Tensor tensor = Tensor::create<ly::LongType>({2, 5}, {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 0
  });

  // cudaMemcpy path (cudnnTransformTensor path is not supported for int16_t)
  Tensor x = F::to(Device::getCuda(), tensor);
  Tensor x2 = F::createTensorLike(x);
  F::copy(x, x2);
  x2 = F::to(Device::getCpu(), x2);

  const ly::LongType *px = tensor.getData<ly::LongType>(), 
                       *pr = x2.getData<ly::LongType>();
  x2.throwIfInvalidShape(tensor.getShape());
  CATCH_REQUIRE(std::equal(px, px + x2.getNumEl(), pr));
}

CATCH_TEST_CASE("test cuda contiguous (copy) 5D", "[cuda][operators][contiguous]") {
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

CATCH_TEST_CASE("test lookup half", "[cuda][operators][lookup]") {
  Tensor embd = F::rand({10, 20}, DType::kFloat);
  Tensor ids = Tensor::create<ly::LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = F::to(Device::getCuda(), embd);
  x = F::cast(x, DType::kFloat16);

  Tensor y = F::to(Device::getCuda(), ids);
  x = F::lookup(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  Tensor xr = F::lookup(embd, ids);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test lookup q4", "[cuda][operators][lookup]") {
  Tensor embd = createRandomQ4Tensor2D(10, 64);
  Tensor ids = Tensor::create<ly::LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = F::to(Device::getCuda(), embd);
  Tensor y = F::to(Device::getCuda(), ids);
  x = F::lookup(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  Tensor xr = F::lookup(embd, ids);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test matmul gemm", "[cuda][operators][matmul]") {
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

CATCH_TEST_CASE("test matmul bmm (gemm)", "[cuda][operators][matmul]") {
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

CATCH_TEST_CASE("test matmul bmm", "[cuda][operators][matmul]") {
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

CATCH_TEST_CASE("test matmul q4", "[cuda][operators][matmul]") {
  Tensor a = F::rand({5, 10, 50}, DType::kFloat);
  Tensor b = createRandomQ4Tensor2D(50, 32);
  Tensor xr = F::matmul(a, b);

  Tensor x = F::to(Device::getCuda(), a);
  Tensor y = F::to(Device::getCuda(), b);
  x = F::cast(x, DType::kFloat16);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test q4 matmul (gemv)", "[cuda][operators][matmul]") {
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

CATCH_TEST_CASE("test scale", "[cuda][operators][scale]") {
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

CATCH_TEST_CASE("test mul", "[cuda][operators][mul]") {
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

CATCH_TEST_CASE("test softmax", "[cuda][operators][softmax]") {
  Tensor a = F::rand({2, 5, 20}, DType::kFloat);
  Tensor xr = F::softmax(a);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = F::softmax(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test add", "[cuda][operators][add]") {
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

CATCH_TEST_CASE("test rms_norm", "[cuda][operators][rms_norm]") {
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

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test causal_mask", "[cuda][operators][causal_mask]") {
  constexpr int DIM = 129;
  Tensor xr = F::softmax(F::causalMask(DIM));
  Tensor x = F::softmax(F::causalMask(DIM, Device::getCuda()));
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}


CATCH_TEST_CASE("test apply_rope", "[cuda][operators][apply_rope]") {
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

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test swiglu", "[cuda][operators][swiglu]") {
  Tensor a = F::rand({2, 10, 16}, DType::kFloat);
  Tensor xr = F::swiglu(a);

  Tensor x = F::to(Device::getCuda(), a);
  x = F::cast(x, DType::kFloat16);
  x = F::swiglu(x);
  x = F::cast(x, DType::kFloat);
  x = F::to(Device::getCpu(), x);

  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test cat", "[cuda][operators][cat]") {
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

CATCH_TEST_CASE("test attention", "[cuda][operators][attention]") {
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

#define CONCAT2(l, r) l ## r
#define CONCAT(l, r) CONCAT2(l, r)

#define LOG_TIME(stmt, message) \
  double CONCAT(t0, __LINE__) = lut::now(); \
  stmt; \
  LOG(INFO) << message <<  ": " << (lut::now() - CONCAT(t0, __LINE__)) * 1000 << "ms";

CATCH_TEST_CASE("benchmark gemv", "[cuda][operators][matmul]") {
  Tensor Aq = createRandomQ4Tensor2D(8000, 4096);
  Tensor x = F::rand({4096, 1}, DType::kFloat);
  Tensor xrq = F::matmul(x.transpose(0, 1), Aq.transpose(0, 1));

  Aq = F::to(Device::getCuda(), Aq);
  x = F::to(Device::getCuda(), x);
  x = F::cast(x, DType::kFloat16);

  Tensor A = ly::op::cuda::dequantQ4ToHalf(Aq);

  LOG_TIME(F::matmul(A, x), "First call F::matmul(A, x)");
  LOG_TIME(Tensor x0 = F::matmul(A, x), "Second call F::matmul(A, x)");
  LOG_TIME(Tensor x1 = ly::op::cuda::gemvHalf(A, x), "ly::op::cuda::gemvHalf(A, x)");
  LOG_TIME(Tensor xq = ly::op::cuda::gemvQ4(Aq, x), "lly::op::cuda::gemvQ4(Aq, x)");

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
