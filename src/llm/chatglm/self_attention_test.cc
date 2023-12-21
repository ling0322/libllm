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
#include "llm/chatglm/self_attention.h"
#include "llm/chatglm/test_common.h"

namespace F = ly::functional;

namespace libllm {
namespace chatglm {

class SelfAttnetionTester : public ly::nn::ModuleTester {
 public:
  SelfAttnetionTester(ly::Device device, ly::DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<SelfAttention> layer = SelfAttention::create(getCtx(), config);
    randomInit(layer);

    ly::Tensor x = generateTensor({1, 20, config.hiddenSize});
    ly::Tensor roPE = generateTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    ly::StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    if (getWeightType() == ly::DType::kQ4) {
      xr0 = {0.9316, -0.0872, -0.5942, -0.4963, -0.0253, -0.0969, -0.4287, 0.3093};
      xr1 = {1.5869, -0.6855, -0.4663, -0.6177, 2.1515e-03, -0.1796, -0.3826, 0.2196};
    } else {
      xr0 = {0.9316, -0.0690, -0.5620, -0.5322, 0.0108, -0.2059, -0.3772, 0.3516};
      xr1 = {1.5160, -0.6959, -0.4921, -0.5905, 0.0328, -0.2900, -0.3377, 0.2644};
    }
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = generateTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x, roPE);
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

CATCH_TEST_CASE("test chatglm::SelfAttnetion", "[llm][chatglm]") {
  SelfAttnetionTester(ly::Device::getCpu(), ly::DType::kFloat).run();
  SelfAttnetionTester(ly::Device::getCpu(), ly::DType::kQ4).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test chatglm::SelfAttnetion (cuda)", "[llm][chatglm][cuda]") {
  SelfAttnetionTester(ly::Device::getCuda(), ly::DType::kFloat).run();
  SelfAttnetionTester(ly::Device::getCuda(), ly::DType::kQ4).run();
}
#endif

}  // namespace chatglm
}  // namespace libllm
