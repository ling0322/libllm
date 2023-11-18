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
