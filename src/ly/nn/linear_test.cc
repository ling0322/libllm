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

#include <array>
#include "ly/ly.h"
#include "ly/nn/test_helper.h"
#include "ly/internal/common.h"
#include "ly/operators/cpu/fingerprint.h"
#include "lyutil/random.h"
#include "lyutil/span.h"

namespace F = ly::functional;

namespace ly {
namespace nn {

class LinearTester : public ModuleTester {
 public:
  static constexpr int InputDim = 128;
  static constexpr int OutputDim = 64;

  LinearTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    std::shared_ptr<Linear> layer = Linear::create(getCtx(), InputDim, OutputDim);
    randomInit(layer);

    Tensor x = randFloatTensor({2, 3, InputDim});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQInt4Group32) {
      xr = {-3.5781, -0.2490, 1.2383, 2.2852, 1.6934, 0.7793, 3.3672, 0.2537};
    } else {
      xr = {-1.5742, -3.1250, 3.6074, -6.2383, -1.4316, 0.4773, 4.7812, -4.0547};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

CATCH_TEST_CASE("test nn::Linear", "[ly][nn][linear]") {
  LinearTester(Device::getCpu(), DType::kFloat).run();
  LinearTester(Device::getCpu(), DType::kQInt4Group32).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test nn::Linear", "[ly][nn][linear][cuda]") {
  LinearTester(Device::getCuda(), DType::kFloat).run();
  LinearTester(Device::getCuda(), DType::kQInt4Group32).run();
}
#endif

}  // namespace nn
}  // namespace ly
