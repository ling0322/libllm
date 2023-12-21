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
#include "llm/chatglm/chatglm_model.h"
#include "llm/chatglm/test_common.h"

namespace F = ly::functional;

namespace libllm {
namespace chatglm {

class ChatGlmTester : public ly::nn::ModuleTester {
 public:
  ChatGlmTester(ly::Device device, ly::DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<ChatGlmModel> layer = ChatGlmModel::create(getCtx(), config);
    randomInit(layer);

    ly::Tensor x = ly::Tensor::create<ly::LongType>({1, 7}, {3, 6, 7, 99, 23, 1, 2});
    x = toTargetDevice(x);

    ly::StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == ly::DType::kQ4) {
      xr0 = {0.0578, -0.2395, -0.4355, 0.0290, 0.0259, 0.4690, -0.5098, 0.4246};
      xr1 = {-0.0422, -0.5171, 0.1490, 0.1388, -0.1149, 0.6445, 0.5312, 0.2837};
    } else {
      xr0 = {0.1182, -0.0625, -0.3939, 0.1281, -0.0249, 0.4091, -0.3467, 0.6563};
      xr1 = {0.1365, -0.3726, 0.2412, 0.2542, -0.1584, 0.6069, 0.7427, 0.3916};
    }
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = ly::Tensor::create<ly::LongType>({1, 1}, {5});
    x = toTargetDevice(x);

    x = layer->forward(past, x);
    CATCH_REQUIRE(allClose(ly::op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

CATCH_TEST_CASE("test chatglm::ChatGlmModel", "[llm][chatglm]") {
  ChatGlmTester(ly::Device::getCpu(), ly::DType::kFloat).run();
  ChatGlmTester(ly::Device::getCpu(), ly::DType::kQ4).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test chatglm::ChatGlmModel (cuda)", "[llm][chatglm][cuda]") {
  ChatGlmTester(ly::Device::getCuda(), ly::DType::kFloat).run();
  ChatGlmTester(ly::Device::getCuda(), ly::DType::kQ4).run();
}
#endif

}  // namespace chatglm
}  // namespace libllm
