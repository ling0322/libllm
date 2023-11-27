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
#include "llyn/device.h"
#include "llyn/llyn.h"
#include "llyn/operators/cpu/cpu_tensor_data.h"
#include "llyn/internal/tensor_shape.h"
#include "lyutil/half.h"
#include "lyutil/random.h"

using llyn::Tensor;
using llyn::DType;
using llyn::Device;
using llyn::internal::TensorShape;
using llyn::op::cpu::CpuTensorData;

namespace F = llyn::functional;

/// @brief Create a random 2D tensor with Q4 type.
Tensor createRandomQ4Tensor2D(int numRow, int numCol) {
  constexpr int groupSize = llyn::QInt4Group32::GroupSize;
  CHECK(numCol % groupSize == 0);
  int numel = numRow * numCol;

  std::vector<uint8_t> data(numel / 2);
  std::vector<float> scaleFloat(numel / groupSize);
  std::vector<uint16_t> scale(numel / groupSize);
  std::vector<int8_t> bias(numel / groupSize);

  int magicNumber = 0x322;
  ly::Random random(magicNumber);
  random.fillUInt8(ly::makeSpan(data));
  random.fillInt8(ly::makeSpan(bias), -2, 18);
  random.fill(ly::makeSpan(scaleFloat), 0, 0.1);
  std::transform(scaleFloat.begin(), scaleFloat.end(), scale.begin(), ly::cvtss_sh);

  auto tensorData = CpuTensorData::create({
      {numel, DType::kQInt4Group32},
      {numel / groupSize, DType::kFloat16},
      {numel / groupSize, DType::kInt8}});
  memcpy(tensorData->getSlot(0)->getRawData(), data.data(), data.size());
  memcpy(tensorData->getSlot(1)->getRawData(), scale.data(), scale.size() * sizeof(uint16_t));
  memcpy(tensorData->getSlot(2)->getRawData(), bias.data(), bias.size());

  std::initializer_list<int> shape{numRow, numCol};
  auto tensorShape = std::make_shared<TensorShape>(shape);

  return Tensor::create(tensorShape, tensorData);
}

CATCH_TEST_CASE("test cuda toDevice", "[cuda][operators][toDevice]") {
  Tensor xCpu = F::rand({100, 200}, DType::kFloat);
  Tensor xCuda = F::toDevice(xCpu, Device(Device::kCuda));
  Tensor xCpu2 = F::toDevice(xCuda, Device(Device::kCpu));
  
  CATCH_REQUIRE(F::allClose(xCpu, xCpu2));
}

CATCH_TEST_CASE("test cuda cast", "[cuda][operators][cast]") {
  Tensor xCpu = F::rand({100, 20, 50}, DType::kFloat);
  Tensor xCuda = F::toDevice(xCpu, Device(Device::kCuda));
  Tensor xHalfCuda = F::cast(xCuda, DType::kFloat16);
  Tensor xCuda2 = F::cast(xHalfCuda, DType::kFloat);
  Tensor xCpu2 = F::toDevice(xCuda2, Device(Device::kCpu));

  CATCH_REQUIRE(F::allClose(xCpu, xCpu2));
}

CATCH_TEST_CASE("test cuda copy", "[cuda][operators][copy]") {
  Tensor tensor = F::rand({100, 20}, DType::kFloat);
  Tensor x = F::toDevice(tensor, Device(Device::kCuda));

  // cudaMemcpy path.
  x = F::cast(x, DType::kFloat16);
  Tensor x2 = F::createTensorLike(x);
  F::copy(x, x2);
  x2 = F::cast(x2, DType::kFloat);
  x2 = F::toDevice(x2, Device(Device::kCpu));
  CATCH_REQUIRE(F::allClose(tensor, x2));

  // cudnnTransformTensor path.
  x = F::toDevice(tensor, Device(Device::kCuda));
  x = F::cast(x, DType::kFloat16);
  x = x.transpose(1, 0);
  x2 = F::createTensorLike(x);
  F::copy(x, x2);

  x2 = F::cast(x2, DType::kFloat);
  x2 = F::toDevice(x2, Device(Device::kCpu));
  CATCH_REQUIRE(F::allClose(tensor, x2.transpose(1, 0)));
}

CATCH_TEST_CASE("test cuda copy (int64_t)", "[cuda][operators][copy]") {
  Tensor tensor = Tensor::create<llyn::LongType>({2, 5}, {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 0
  });

  // cudaMemcpy path (cudnnTransformTensor path is not supported for int16_t)
  Tensor x = F::toDevice(tensor, Device(Device::kCuda));
  Tensor x2 = F::createTensorLike(x);
  F::copy(x, x2);
  x2 = F::toDevice(x2, Device(Device::kCpu));

  const llyn::LongType *px = tensor.getData<llyn::LongType>(), 
                       *pr = x2.getData<llyn::LongType>();
  x2.throwIfInvalidShape(tensor.getShape());
  CATCH_REQUIRE(std::equal(px, px + x2.getNumEl(), pr));
}

CATCH_TEST_CASE("test cuda contiguous (copy) 5D", "[cuda][operators][contiguous]") {
  Tensor tensor = F::rand({10, 2, 5, 20}, DType::kFloat);

  Tensor x = F::toDevice(tensor, Device(Device::kCuda));
  x = F::cast(x, DType::kFloat16);
  x = x.unsqueeze(1).expand({10, 4, 2, 5, 20});
  x = F::contiguous(x);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(x, Device(Device::kCpu));

  Tensor xr = tensor.unsqueeze(1).expand({10, 4, 2, 5, 20});
  xr = F::contiguous(x);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test lookup half", "[cuda][operators][lookup]") {
  Tensor embd = F::rand({10, 20}, DType::kFloat);
  Tensor ids = Tensor::create<llyn::LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = F::toDevice(embd, Device(Device::kCuda));
  x = F::cast(x, DType::kFloat16);

  Tensor y = F::toDevice(ids, Device(Device::kCuda));
  x = F::lookup(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(x, Device(Device::kCpu));

  Tensor xr = F::lookup(embd, ids);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test lookup q4", "[cuda][operators][lookup]") {
  Tensor embd = createRandomQ4Tensor2D(10, 64);
  Tensor ids = Tensor::create<llyn::LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = F::toDevice(embd, Device(Device::kCuda));
  Tensor y = F::toDevice(ids, Device(Device::kCuda));
  x = F::lookup(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(x, Device(Device::kCpu));

  Tensor xr = F::lookup(embd, ids);
  CATCH_REQUIRE(F::allClose(x, xr));
}

CATCH_TEST_CASE("test matmul gemm", "[cuda][operators][matmul]") {
  Tensor a = F::rand({10, 20}, DType::kFloat);
  Tensor b = F::rand({40, 30}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(1, 0));

  Tensor x = F::toDevice(a, Device(Device::kCuda));
  Tensor y = F::toDevice(b, Device(Device::kCuda));
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(1, 0);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(x, Device(Device::kCpu));

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test matmul bmm (gemm)", "[cuda][operators][matmul]") {
  Tensor a = F::rand({5, 10, 20}, DType::kFloat);
  Tensor b = F::rand({40, 30}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(1, 0));

  Tensor x = F::toDevice(a, Device(Device::kCuda));
  Tensor y = F::toDevice(b, Device(Device::kCuda));
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(1, 0);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(x, Device(Device::kCpu));

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}

CATCH_TEST_CASE("test matmul bmm", "[cuda][operators][matmul]") {
  Tensor a = F::rand({5, 10, 5, 20}, DType::kFloat);
  Tensor b = F::rand({10, 30, 20}, DType::kFloat);
  Tensor xr = F::matmul(a, b.slice(1, {5, 25}).transpose(-1, -2));

  Tensor x = F::toDevice(a, Device(Device::kCuda));
  Tensor y = F::toDevice(b, Device(Device::kCuda));
  x = F::cast(x, DType::kFloat16);
  y = F::cast(y, DType::kFloat16);
  y = y.slice(1, {5, 25});
  y = y.transpose(-1, -2);
  x = F::matmul(x, y);
  x = F::cast(x, DType::kFloat);
  x = F::toDevice(x, Device(Device::kCpu));

  CATCH_REQUIRE(F::allClose(x, xr, 1e-5f, 2e-2f));
}
