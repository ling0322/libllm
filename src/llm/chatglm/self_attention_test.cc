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

    ly::Tensor x = randFloatTensor({1, 20, config.hiddenSize});
    ly::Tensor roPE = randFloatTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    ly::StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    if (getWeightType() == ly::DType::kQInt4Group32) {
      xr0 = {0.0450, 0.3328, 0.0659, 0.1888, 0.1116, 0.2878, 0.5171, -0.4473};
      xr1 = {0.2280, 0.2035, 0.2007, 0.1946, 0.0179, 0.2151, 0.2145, -0.2549};
    } else {
      xr0 = {0.8604, -0.0639, -0.5186, -0.4915, 9.9182e-03, -0.1903, -0.3486, 0.3245};
      xr1 = {1.4004, -0.6431, -0.4546, -0.5449, 0.0305, -0.2681, -0.3118, 0.2439};
    }
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = randFloatTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x, roPE);
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

CATCH_TEST_CASE("test chatglm::SelfAttnetion", "[llm][chatglm]") {
  SelfAttnetionTester(ly::Device::getCpu(), ly::DType::kFloat).run();
  SelfAttnetionTester(ly::Device::getCpu(), ly::DType::kQInt4Group32).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test chatglm::SelfAttnetion (cuda)", "[llm][chatglm][cuda]") {
  SelfAttnetionTester(ly::Device::getCuda(), ly::DType::kFloat).run();
  SelfAttnetionTester(ly::Device::getCuda(), ly::DType::kQInt4Group32).run();
}
#endif

}  // namespace chatglm
}  // namespace libllm
