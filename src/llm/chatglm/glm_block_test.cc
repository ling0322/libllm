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
#include "llm/chatglm/glm_block.h"
#include "llm/chatglm/test_common.h"

namespace F = ly::functional;

namespace libllm {
namespace chatglm {

class GlmBlockTester : public ly::nn::ModuleTester {
 public:
  GlmBlockTester(ly::Device device, ly::DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<GLMBlock> layer = GLMBlock::create(getCtx(), config);
    randomInit(layer);

    ly::Tensor x = generateTensor({1, 20, config.hiddenSize});
    ly::Tensor roPE = generateTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    ly::StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    if (getWeightType() == ly::DType::kQInt4Group32) {
      xr0 = {0.5889, -2.1953, 0.0465, -0.1042, 0.0484, -0.0375, -0.1042, 0.8555};
      xr1 = {1.2363, -1.1816, 0.3521, -1.3564, 0.7534, 0.1021, -0.4507, 0.3972};
    } else {
      xr0 = {0.4910, -2.1172, 0.0845, -0.0580, 0.0892, -0.1219, -0.1042, 0.8716};
      xr1 = {1.1953, -1.1631, 0.3333, -1.3271, 0.8140, -0.0444, -0.4258, 0.4353};
    }
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = generateTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x, roPE);
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

CATCH_TEST_CASE("test chatglm::GLMBlock", "[llm][chatglm]") {
  GlmBlockTester(ly::Device::getCpu(), ly::DType::kFloat).run();
  GlmBlockTester(ly::Device::getCpu(), ly::DType::kQInt4Group32).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test chatglm::GLMBlock (cuda)", "[llm][chatglm][cuda]") {
  GlmBlockTester(ly::Device::getCuda(), ly::DType::kFloat).run();
  GlmBlockTester(ly::Device::getCuda(), ly::DType::kQInt4Group32).run();
}
#endif

}  // namespace chatglm
}  // namespace libllm
