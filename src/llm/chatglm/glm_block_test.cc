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

    ly::Tensor x = randFloatTensor({1, 20, config.hiddenSize});
    ly::Tensor roPE = randFloatTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    ly::StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    float absTol;
    if (getWeightType() == ly::DType::kQInt4Group32) {
      xr0 = {1.4526, 4.6741, -3.3150, 1.2496, 0.3357, 4.4487, 5.4043, -0.3533};
      xr1 = {6.1540, -4.4297, -5.2617, 12.8565, 2.5964, 13.4825, 2.6722, 1.9815};
      absTol = 0.01;
    } else {
      xr0 = {0.4840, -3.1771, -3.0471, 4.1138, -1.9824, -0.1645, -0.6486, 2.1234};
      xr1 = {1.2939, -3.4805, 0.2722, 3.6699, -4.7578, 2.1973, -2.5000, 7.4414};
      absTol = 0.2;
    }
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr0, absTol));

    // forward next token.
    x = randFloatTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x, roPE);
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr1, absTol));
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
