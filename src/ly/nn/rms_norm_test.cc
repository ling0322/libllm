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

#include "ly/ly.h"
#include "ly/internal/common.h"
#include "ly/operators/cpu/fingerprint.h"
#include "lyutil/random.h"

namespace F = ly::functional;

namespace ly {
namespace nn {

void testRmsNorm(Device device, DType quantType, Tensor fingerprint) {
  constexpr int DIM = 512;
  lut::Random rand(internal::RandomSeed);

  Context ctx;
  ctx.setDevice(device);
  ctx.setFloatDType(F::getDefaultFloatType(device));
  std::shared_ptr<RMSNorm> layer = RMSNorm::create(ctx, DIM, 1e-5);
  layer->initParameters(&rand, quantType);

  Tensor x = F::rand({2, DIM}, DType::kFloat, Device::getCpu(), &rand);
  x = layer->forward(x);

  CATCH_REQUIRE(F::allClose(op::cpu::fingerprint(x), fingerprint));
}

CATCH_TEST_CASE("test nn::RmsNorm", "[ly][nn][rms_norm]") {
  testRmsNorm(Device::getCpu(), DType::kFloat, Tensor::create<float>({8}, {
    -0.0659, -0.8178, -0.1703, 0.0363, 0.9171, -0.3605, -0.3002, -0.1001
  }));
}


}  // namespace nn
}  // namespace ly
