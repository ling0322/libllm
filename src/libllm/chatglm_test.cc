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

#include "libllm/chatglm.h"

#include <array>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/cpu/fingerprint.h"
#include "libllm/lut/random.h"
#include "libllm/lut/span.h"
#include "libllm/tensor.h"
#include "libllm/test_helper.h"

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
  ChatGlmTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<ChatGlmModel> layer = ChatGlmModel::create(getCtx(), config);
    randomInit(layer);

    Tensor x = Tensor::create<LongType>({1, 7}, {3, 6, 7, 99, 23, 1, 2});
    x = toTargetDevice(x);

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {0.1738, 1.0190, -0.4752, 0.2405, 0.0359, 0.0821, -0.3922, 0.7598};
      xr1 = {-0.0869, 0.3474, 0.3587, 0.3405, -0.1254, 0.4693, 1.6388, 0.4857};
    } else {
      xr0 = {0.1209, 1.0635, -0.3218, 0.2810, 0.0438, 0.0445, -0.4287, 0.7803};
      xr1 = {-0.1908, 0.2644, 0.4468, 0.3596, -0.1265, 0.4871, 1.4609, 0.5425};
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
  GlmBlockTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<GLMBlock> layer = GLMBlock::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {-0.6995, -1.7179, 0.6030, 0.4212, 0.1758, -0.0991, 0.1561, 0.3834};
      xr1 = {0.0515, -0.5976, 0.9120, -0.8255, 0.5175, 0.3463, 0.1092, 0.0433};
    } else {
      xr0 = {-0.6816, -1.6572, 0.6406, 0.4470, 0.1726, -0.1067, 0.1279, 0.3716};
      xr1 = {0.0298, -0.5806, 0.9023, -0.8027, 0.5244, 0.2893, 0.0976, 0.0643};
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
  MlpTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<MLP> layer = MLP::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQInt4x32) {
      xr = {6.4475e-03, -0.1200, -0.1880, 0.1787, -0.0284, -0.3479, 0.0240, 0.1778};
    } else {
      xr = {-4.1466e-03, -0.1296, -0.1753, 0.1750, -0.0217, -0.3584, -5.1689e-03, 0.1554};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

class SelfAttnetionTester : public ModuleTester {
 public:
  SelfAttnetionTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  float getRtol() const override {
    DType defaultFloatType = F::getDefaultFloatType(getDevice());
    if (getDevice().getType() == Device::kCpu && defaultFloatType == DType::kFloat16) {
      return 3.5e-2;
    } else {
      return 5e-3;
    }
  }

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<SelfAttention> layer = SelfAttention::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSizePerAttentionHead / 2});

    StateMap past;
    x = layer->forward(past, x, roPE);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {-0.4076, 0.4038, -0.1918, -9.9697e-03, -6.4159e-03, 0.0496, -0.0568, 0.0421};
      xr1 = {0.3660, -0.1488, -0.0495, -0.0875, 0.0862, -0.0287, -9.2791e-03, -0.0864};
    } else {
      xr0 = {-0.3489, 0.4617, -0.1710, -0.0709, 0.0242, 0.0579, -0.0675, 0.0418};
      xr1 = {0.3424, -0.1663, -0.0594, -0.0924, 0.0969, -0.0248, -5.9996e-03, -0.0793};
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
  ChatGlmTester(Device::getCpu(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test chatglm::GLMBlock", "[llm][chatglm]") {
  GlmBlockTester(Device::getCpu(), DType::kFloat).run();
  GlmBlockTester(Device::getCpu(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test chatglm::MLP", "[llm][chatglm]") {
  MlpTester(Device::getCpu(), DType::kFloat).run();
  MlpTester(Device::getCpu(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test chatglm::SelfAttnetion", "[llm][chatglm]") {
  SelfAttnetionTester(Device::getCpu(), DType::kFloat).run();
  SelfAttnetionTester(Device::getCpu(), DType::kQInt4x32).run();
}

#ifdef LIBLLM_CUDA_ENABLED
CATCH_TEST_CASE("test chatglm::ChatGlmModel (cuda)", "[llm][chatglm][cuda]") {
  ChatGlmTester(Device::getCuda(), DType::kFloat).run();
  ChatGlmTester(Device::getCuda(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test chatglm::GLMBlock (cuda)", "[llm][chatglm][cuda]") {
  GlmBlockTester(Device::getCuda(), DType::kFloat).run();
  GlmBlockTester(Device::getCuda(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test chatglm::MLP (cuda)", "[llm][chatglm][cuda]") {
  MlpTester(Device::getCuda(), DType::kFloat).run();
  MlpTester(Device::getCuda(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test chatglm::SelfAttnetion (cuda)", "[llm][chatglm][cuda]") {
  SelfAttnetionTester(Device::getCuda(), DType::kFloat).run();
  SelfAttnetionTester(Device::getCuda(), DType::kQInt4x32).run();
}
#endif

}  // namespace chatglm
}  // namespace libllm
