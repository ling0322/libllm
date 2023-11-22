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

#include "../../../third_party/catch2/catch_amalgamated.hpp"

#include <algorithm>
#include "llyn/device.h"
#include "llyn/llyn.h"

int main(int argc, char **argv) {
  llyn::init();

  int result = Catch::Session().run(argc, argv);
  llyn::destroy();

  return result;
}

using llyn::Tensor;
using llyn::DType;
using llyn::Device;

namespace F = llyn::functional;

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
