// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "libllm/llama.h"

#include <array>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/cpu/fingerprint.h"
#include "libllm/lut/random.h"
#include "libllm/lut/span.h"
#include "libllm/tensor.h"
#include "libllm/test_helper.h"

namespace libllm {
namespace llama {

class TestCommon {
 public:
  static LlamaConfig getConfig() {
    LlamaConfig config;
    config.hiddenSize = 128;
    config.numHeads = 2;
    config.numKeyValueHeads = 2;
    config.intermediateSize = 256;
    config.normEps = 1e-5;
    config.numLayers = 2;
    config.vocabSize = 100;
    config.maxContextLength = 200;
    config.qkvProjBias = false;

    return config;
  }
};

class LlamaTester : public ModuleTester {
 public:
  LlamaTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  float getRtol() const override {
    DType defaultFloatType = F::getDefaultFloatType(getDevice());
    if (getDevice().getType() == Device::kCpu && defaultFloatType == DType::kFloat16) {
      return 3e-2;
    } else {
      return 5e-3;
    }
  }

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<LlamaModel> layer = LlamaModel::create(getCtx(), config);
    randomInit(layer);

    Tensor x = Tensor::create<LongType>({1, 7}, {3, 6, 7, 99, 23, 1, 2});
    x = toTargetDevice(x);

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {-0.3387, -0.0333, 0.0540, -0.0398, -0.3014, -0.0286, -0.5364, 0.7264};
      xr1 = {-0.4566, -0.1807, -0.1087, -0.2393, 0.1119, 0.0876, 0.4581, 0.9707};
    } else {
      xr0 = {-0.3502, -0.0495, 0.0601, 6.2854e-03, -0.3113, -0.0240, -0.5557, 0.7237};
      xr1 = {-0.4593, -0.1671, -0.1131, -0.2362, 0.1557, 0.1016, 0.4251, 0.9781};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = Tensor::create<LongType>({1, 1}, {5});
    x = toTargetDevice(x);

    x = layer->forward(past, x);
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

class DecoderLayerTester : public ModuleTester {
 public:
  DecoderLayerTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<DecodeLayer> layer = DecodeLayer::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSize / config.numHeads / 2});

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {0.3131, 0.4854, 1.3699, 0.1522, -0.3090, 0.6377, 0.8493, -0.0997};
      xr1 = {-0.0261, -0.2735, -0.6873, -1.1042, -0.1448, -1.0856, 0.8208, 0.5531};
    } else {
      xr0 = {0.3320, 0.4682, 1.3183, 0.1654, -0.3265, 0.6601, 0.8897, -0.1111};
      xr1 = {-0.0557, -0.2677, -0.6885, -1.1301, -0.1634, -1.0932, 0.8492, 0.5416};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = generateTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x);
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

class MlpTester : public ModuleTester {
 public:
  MlpTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<MLP> layer = MLP::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQInt4x32) {
      xr = {-0.0910, 0.0423, -0.1808, -0.1989, -0.3223, -0.2231, 0.0283, -0.1616};
    } else {
      xr = {-0.0846, 0.0652, -0.1784, -0.1916, -0.3176, -0.1829, 0.0203, -0.1310};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

class AttnetionTester : public ModuleTester {
 public:
  AttnetionTester(Device device, DType weightType)
      : ModuleTester(device, weightType) {
  }

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<Attention> layer = Attention::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSize / config.numHeads / 2});

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQInt4x32) {
      xr0 = {-0.4392, 0.2154, 0.2774, -0.0109, 0.0226, -0.0604, -0.0747, -0.0398};
      xr1 = {0.0673, 0.0424, 0.1065, -0.1059, 0.0920, -0.0598, -0.0223, 1.8101e-03};
    } else {
      xr0 = {-0.4450, 0.1851, 0.2650, -1.4753e-03, 0.0306, -0.0671, -0.0855, -0.0340};
      xr1 = {0.0567, 0.0309, 0.1073, -0.0987, 0.0984, -0.0587, -0.0246, 4.9177e-03};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr0));

    // forward next token.
    x = generateTensor({1, 1, config.hiddenSize});
    x = layer->forward(past, x);
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr1));
  }
};

CATCH_TEST_CASE("test llama::LlamaModel", "[llm][llama]") {
  LlamaTester(Device::getCpu(), DType::kFloat).run();
  LlamaTester(Device::getCpu(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test llama::DecoderLayer", "[llm][llama]") {
  DecoderLayerTester(Device::getCpu(), DType::kFloat).run();
  DecoderLayerTester(Device::getCpu(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test llama::MLP", "[llm][llama]") {
  MlpTester(Device::getCpu(), DType::kFloat).run();
  MlpTester(Device::getCpu(), DType::kQInt4x32).run();
}

CATCH_TEST_CASE("test llama::Attnetion", "[llm][llama]") {
  AttnetionTester(Device::getCpu(), DType::kFloat).run();
  AttnetionTester(Device::getCpu(), DType::kQInt4x32).run();
}

}  // namespace llama
}  // namespace libllm
