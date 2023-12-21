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
#include "llm/chatglm/mlp.h"
#include "llm/chatglm/test_common.h"

namespace F = ly::functional;

namespace libllm {
namespace chatglm {

class MlpTester : public ly::nn::ModuleTester {
 public:
  MlpTester(ly::Device device, ly::DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<MLP> layer = MLP::create(getCtx(), config);
    randomInit(layer);

    ly::Tensor x = generateTensor({1, 20, config.hiddenSize});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == ly::DType::kQ4) {
      xr = {0.0239, -0.0499, -0.0876, 0.0995, -0.0488, -0.1304, 0.0221, 0.0688};
    } else {
      xr = {-0.0114, -0.0648, -0.0796, 0.0816, -0.0117, -0.1505, 1.0281e-03, 0.0742};
    }
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr));
  }
};

CATCH_TEST_CASE("test chatglm::MLP", "[llm][chatglm]") {
  MlpTester(ly::Device::getCpu(), ly::DType::kFloat).run();
  MlpTester(ly::Device::getCpu(), ly::DType::kQ4).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test chatglm::MLP (cuda)", "[llm][chatglm][cuda]") {
  MlpTester(ly::Device::getCuda(), ly::DType::kFloat).run();
  MlpTester(ly::Device::getCuda(), ly::DType::kQ4).run();
}
#endif

}  // namespace chatglm
}  // namespace libllm
