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

#include "catch2/catch_amalgamated.hpp"

#include <array>
#include "libllm/tensor.h"
#include "libllm/test_helper.h"
#include "libllm/cpu/fingerprint.h"
#include "libllm/lut/random.h"
#include "libllm/lut/span.h"
#include "libllm/chatglm.h"

namespace libllm {
namespace chatglm {

class TestCommon {
 public:
  static ChatGlmConfig getConfig() {
    ChatGlmConfig config;
    config.ffnHiddenSize = 512;
    config.hiddenSize = 256;
    config.hiddenSizePerAttentionHead = 64;
    config.kvChannels = 64;
    config.multiQueryGroupNum = 2;
    config.normEps = 1e-5;
    config.numLayers = 2;
    config.seqLength = 8192;
    config.symbolEOS = 2;
    config.symbolGMask = 98;
    config.symbolSOP = 99;
    config.vocabSize = 100;

    return config;
  }
};

class ChatGlmTester : public ModuleTester {
 public:
  ChatGlmTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<ChatGlmModel> layer = ChatGlmModel::create(getCtx(), config);
    randomInit(layer);

    Tensor x = Tensor::create<LongType>({1, 7}, {3, 6, 7, 99, 23, 1, 2});
    x = toTargetDevice(x);

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQ4) {
      xr0 = {0.0578, -0.2395, -0.4355, 0.0290, 0.0259, 0.4690, -0.5098, 0.4246};
      xr1 = {-0.0422, -0.5171, 0.1490, 0.1388, -0.1149, 0.6445, 0.5312, 0.2837};
    } else {
      xr0 = {0.1182, -0.0625, -0.3939, 0.1281, -0.0249, 0.4091, -0.3467, 0.6563};
      xr1 = {0.1365, -0.3726, 0.2412, 0.2542, -0.1584, 0.6069, 0.7427, 0.3916};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = Tensor::create<LongType>({1, 1}, {5});
    x = toTargetDevice(x);

    x = layer->forward(past, x);
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

class GlmBlockTester : public ModuleTester {
 public:
  GlmBlockTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<GLMBlock> layer = GLMBlock::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQ4) {
      xr0 = {0.2781, -2.0898, 0.1926, -0.0876, 0.0167, -0.1566, -0.1160, 0.8701};
      xr1 = {1.1748, -1.2783, 0.3721, -1.3730, 0.7666, -0.1094, -0.4922, 0.4146};
    } else {
      xr0 = {0.4910, -2.1172, 0.0845, -0.0580, 0.0892, -0.1219, -0.1042, 0.8716};
      xr1 = {1.1953, -1.1631, 0.3333, -1.3271, 0.8140, -0.0444, -0.4258, 0.4353};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = generateTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x, roPE);
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

class MlpTester : public ModuleTester {
 public:
  MlpTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<MLP> layer = MLP::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQ4) {
      xr = {0.0239, -0.0499, -0.0876, 0.0995, -0.0488, -0.1304, 0.0221, 0.0688};
    } else {
      xr = {-0.0114, -0.0648, -0.0796, 0.0816, -0.0117, -0.1505, 1.0281e-03, 0.0742};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

class SelfAttnetionTester : public ModuleTester {
 public:
  SelfAttnetionTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<SelfAttention> layer = SelfAttention::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQ4) {
      xr0 = {0.7993, -0.1409, -0.4656, -0.6050, -0.0488, -0.2223, -0.4678, 0.2935};
      xr1 = {1.5264, -0.8032, -0.4797, -0.6748, -0.0331, -0.3162, -0.4333, 0.2161};
    } else {
      xr0 = {0.9316, -0.0690, -0.5620, -0.5322, 0.0108, -0.2059, -0.3772, 0.3516};
      xr1 = {1.5160, -0.6959, -0.4921, -0.5905, 0.0328, -0.2900, -0.3377, 0.2644};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = generateTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x, roPE);
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

CATCH_TEST_CASE("test chatglm::ChatGlmModel", "[llm][chatglm]") {
  ChatGlmTester(Device::getCpu(), DType::kFloat).run();
  ChatGlmTester(Device::getCpu(), DType::kQ4).run();
}

CATCH_TEST_CASE("test chatglm::GLMBlock", "[llm][chatglm]") {
  GlmBlockTester(Device::getCpu(), DType::kFloat).run();
  GlmBlockTester(Device::getCpu(), DType::kQ4).run();
}

CATCH_TEST_CASE("test chatglm::MLP", "[llm][chatglm]") {
  MlpTester(Device::getCpu(), DType::kFloat).run();
  MlpTester(Device::getCpu(), DType::kQ4).run();
}

CATCH_TEST_CASE("test chatglm::SelfAttnetion", "[llm][chatglm]") {
  SelfAttnetionTester(Device::getCpu(), DType::kFloat).run();
  SelfAttnetionTester(Device::getCpu(), DType::kQ4).run();
}

#ifdef LIBLLM_CUDA_ENABLED
CATCH_TEST_CASE("test chatglm::ChatGlmModel (cuda)", "[llm][chatglm][cuda]") {
  ChatGlmTester(Device::getCuda(), DType::kFloat).run();
  ChatGlmTester(Device::getCuda(), DType::kQ4).run();
}

CATCH_TEST_CASE("test chatglm::GLMBlock (cuda)", "[llm][chatglm][cuda]") {
  GlmBlockTester(Device::getCuda(), DType::kFloat).run();
  GlmBlockTester(Device::getCuda(), DType::kQ4).run();
}

CATCH_TEST_CASE("test chatglm::MLP (cuda)", "[llm][chatglm][cuda]") {
  MlpTester(Device::getCuda(), DType::kFloat).run();
  MlpTester(Device::getCuda(), DType::kQ4).run();
}

CATCH_TEST_CASE("test chatglm::SelfAttnetion (cuda)", "[llm][chatglm][cuda]") {
  SelfAttnetionTester(Device::getCuda(), DType::kFloat).run();
  SelfAttnetionTester(Device::getCuda(), DType::kQ4).run();
}
#endif

}  // namespace chatglm
}  // namespace libllm
