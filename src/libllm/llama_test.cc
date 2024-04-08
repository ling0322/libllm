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

#include "catch2/catch_amalgamated.hpp"

#include <array>
#include "libllm/tensor.h"
#include "libllm/test_helper.h"
#include "libllm/cpu/fingerprint.h"
#include "libllm/lut/random.h"
#include "libllm/lut/span.h"
#include "libllm/llama.h"

namespace libllm {
namespace llama {

class TestCommon {
 public:
  static LlamaConfig getConfig() {
    LlamaConfig config;
    config.hiddenSize = 128;
    config.numHeads = 2;
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
  LlamaTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<LlamaModel> layer = LlamaModel::create(getCtx(), config);
    randomInit(layer);

    Tensor x = Tensor::create<LongType>({1, 7}, {3, 6, 7, 99, 23, 1, 2});
    x = toTargetDevice(x);

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQ4) {
      xr0 = {-0.1072, -0.2913, 0.0412, -0.1886, -0.0395, 8.2878e-03, 0.2479, -0.6602};
      xr1 = {-0.4707, -3.1668e-04, -0.0181, -0.4168, 0.5044, 0.0581, 0.1294, -0.2918};
    } else {
      xr0 = {-0.1778, -0.1569, 0.0550, -0.1257, 0.0480, 3.1704e-03, 0.2766, -0.589};
      xr1 = {-0.4089, 0.1488, -0.0555, -0.4014, 0.5508, 0.0278, 0.1433, -0.1969};
    }
    LOG(INFO) << "x vs xr0";
    F::print(op::cpu::fingerprint(toCpu(x)));
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
  DecoderLayerTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<DecodeLayer> layer = DecodeLayer::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSize / config.numHeads / 2});

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQ4) {
      xr0 = {-0.8978, -0.1241, 2.8211, 0.8802, -1.3381, 0.2263, 1.3836, 1.7381};
      xr1 = {0.5589, -0.1282, -2.1789, -2.0911, -0.4496, -1.8510, 0.6279, 1.0239};
    } else {
      xr0 = {-0.7904, 0.0163, 3.5986, -0.1162, -1.4910, 0.1865, 1.3464, 1.3563};
      xr1 = {0.5005, 0.0787, -2.0844, -2.2465, -0.6049, -1.8843, 0.7063, 0.7963};
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
  MlpTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<MLP> layer = MLP::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQ4) {
      xr = {-0.7585, 0.6718, -0.7877, -0.9301, -1.5713, -1.3301, 0.1308, -0.9319};
    } else {
      xr = {-0.5706, 0.3436, -0.9281, -1.1502, -1.9871, -1.1115, 0.1469, -0.9078};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

class AttnetionTester : public ModuleTester {
 public:
  AttnetionTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    LlamaConfig config = TestCommon::getConfig();
    std::shared_ptr<Attention> layer = Attention::create(getCtx(), config);
    randomInit(layer);

    Tensor x = generateTensor({1, 20, config.hiddenSize});
    Tensor roPE = generateTensor({256, 1, config.hiddenSize / config.numHeads / 2});

    StateMap past;
    x = layer->forward(past, x);

    std::vector<float> xr0, xr1;
    if (getWeightType() == DType::kQ4) {
      xr0 = {-1.1863, 0.6394, 0.7961, -0.3720, 0.0420, -0.2304, -0.1991, -0.1881};
      xr1 = {0.1828, 0.0831, 0.2031, -0.3235, 0.1687, -0.1683, -0.1076, -0.0304};
    } else {
      xr0 = {-1.3349, 0.7137, 0.8068, -0.1421, 0.0925, -0.2039, -0.2464, -0.0960};
      xr1 = {0.2007, 0.0961, 0.2528, -0.3295, 0.2023, -0.1546, -0.1257, 0.0487};
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
  LlamaTester(Device::getCpu(), DType::kQ4).run();
}

CATCH_TEST_CASE("test llama::DecoderLayer", "[llm][llama]") {
  DecoderLayerTester(Device::getCpu(), DType::kFloat).run();
  DecoderLayerTester(Device::getCpu(), DType::kQ4).run();
}

CATCH_TEST_CASE("test llama::MLP", "[llm][llama]") {
  MlpTester(Device::getCpu(), DType::kFloat).run();
  MlpTester(Device::getCpu(), DType::kQ4).run();
}

CATCH_TEST_CASE("test llama::Attnetion", "[llm][llama]") {
  AttnetionTester(Device::getCpu(), DType::kFloat).run();
  AttnetionTester(Device::getCpu(), DType::kQ4).run();
}

}  // namespace llama
}  // namespace libllm
