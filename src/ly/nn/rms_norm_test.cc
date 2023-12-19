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
#include "ly/operators/cpu/fingerprint.h"
#include "lyutil/random.h"
#include "lyutil/span.h"

namespace F = ly::functional;

namespace ly {
namespace nn {

class RmsNormTester : public ModuleTester {
 public:
  static constexpr int FeatureDim = 512;

  RmsNormTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    std::shared_ptr<RMSNorm> layer = RMSNorm::create(getCtx(), FeatureDim, 1e-5);
    randomInit(layer);

    Tensor x = generateTensor({2, 3, FeatureDim});
    x = layer->forward(x);

    std::vector<float> xr = {-0.0659, 0.6089, -0.2864, -0.3371, 0.9171, -0.3605, -1.1276, 1.0056};
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

CATCH_TEST_CASE("test nn::RmsNorm", "[ly][nn][rms_norm]") {
  RmsNormTester(Device::getCpu(), DType::kFloat).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test nn::RmsNorm", "[ly][nn][rms_norm][cuda]") {
  RmsNormTester(Device::getCuda(), DType::kFloat).run();
}
#endif

}  // namespace nn
}  // namespace ly
