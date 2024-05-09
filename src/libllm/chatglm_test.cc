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
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {0.1227, 0.9854, -0.3079, 0.2357, 0.0835, 0.1189, -0.6709, 0.8042};
      xr1 = {-0.5444, 0.4097, 0.5439, 0.2766, -0.0801, 0.4790, 1.4492, 0.6235};
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
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {-0.8677, -1.4365, 0.7910, 0.4990, 0.1694, -0.1394, 0.1875, 0.4575};
      xr1 = {0.0284, -0.5552, 0.9102, -0.7754, 0.5430, 0.2642, 0.0641, 0.0549};
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
  MlpTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    ChatGlmConfig config = TestCommon::getConfig();
    std::shared_ptr<MLP> layer = MLP::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQInt4x32) {
      xr = {0.0764, -0.1021, -0.1920, 0.2137, -0.1087, -0.3198, 0.0374, 0.1409};
    } else {
      xr = {-4.1466e-03, -0.1296, -0.1753, 0.1750, -0.0217, -0.3584, -5.1689e-03, 0.1554};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

class SelfAttnetionTester : public ModuleTester {
 public:
  SelfAttnetionTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

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
      xr0 = {-0.4969, 0.4878, -0.0815, -0.0862, 0.0241, 0.0619, -0.1013, 0.0136};
      xr1 = {0.3715, -0.1946, -0.0598, -0.1155, 0.0750, -0.0403, -0.0444, -0.0987};
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
