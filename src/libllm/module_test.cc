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


namespace libllm {

class LinearTester : public ModuleTester {
 public:
  static constexpr int InputDim = 128;
  static constexpr int OutputDim = 64;

  LinearTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    std::shared_ptr<Linear> layer = Linear::create(getCtx(), InputDim, OutputDim);
    randomInit(layer);

    Tensor x = generateTensor({2, 3, InputDim});
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQ4) {
      xr = {-0.4941, -1.0957, 0.3513, -0.8105, -0.9609, 0.3569, 0.3374, 0.0200};
    } else {
      xr = {-0.4829, -1.0068, 0.3118, -0.7295, -1.0000, 0.4971, 0.3013, 0.0405};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

class RmsNormTester : public ModuleTester {
 public:
  static constexpr int FeatureDim = 512;

  RmsNormTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    std::shared_ptr<RMSNorm> layer = RMSNorm::create(getCtx(), FeatureDim, 1e-5);
    randomInit(layer);

    Tensor x = generateTensor({2, 3, FeatureDim});
    x = layer->forward(x);

    std::vector<float> xr = {-0.0659, 0.6089, -0.2864, -0.3371, 0.9171, -0.3605, -1.1276, 1.0056};
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

class EmbeddingTester : public ModuleTester {
 public:
  static constexpr int HiddenSize = 128;
  static constexpr int VocabSize = 64;

  EmbeddingTester(Device device, DType weightType) : ModuleTester(device, weightType) {}

  void run() {
    std::shared_ptr<Embedding> layer = Embedding::create(getCtx(), HiddenSize, VocabSize);
    randomInit(layer);

    Tensor x = Tensor::create<LongType>({1, 8}, {1, 2, 5, 9, 19, 23, 29, 63});
    x = toTargetDevice(x);
    x = layer->forward(x);

    std::vector<float> xr;
    if (getWeightType() == DType::kQ4) {
      xr = {-0.9160, 0.6548, -0.9067, -0.1317, -0.5322, 0.7900, -1.0605, 0.0};
    } else {
      xr = {-0.8596, 0.7052, -0.9062, -0.2155, -0.5044, 0.7694, -0.9440, -0.0745};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

CATCH_TEST_CASE("test Linear", "[ly][nn][linear]") {
  LinearTester(Device::getCpu(), DType::kFloat).run();
  LinearTester(Device::getCpu(), DType::kQ4).run();
}

CATCH_TEST_CASE("test RmsNorm", "[ly][nn][rms_norm]") {
  RmsNormTester(Device::getCpu(), DType::kFloat).run();
}

CATCH_TEST_CASE("test Embedding", "[ly][nn][embedding]") {
  EmbeddingTester(Device::getCpu(), DType::kFloat).run();
  EmbeddingTester(Device::getCpu(), DType::kQ4).run();
}

#ifdef LIBLLM_CUDA_ENABLED
CATCH_TEST_CASE("test Linear", "[ly][nn][linear][cuda]") {
  LinearTester(Device::getCuda(), DType::kFloat).run();
  LinearTester(Device::getCuda(), DType::kQ4).run();
}

CATCH_TEST_CASE("test RmsNorm", "[ly][nn][rms_norm][cuda]") {
  RmsNormTester(Device::getCuda(), DType::kFloat).run();
}

CATCH_TEST_CASE("test Embedding (cuda)", "[ly][nn][embedding][cuda]") {
  EmbeddingTester(Device::getCuda(), DType::kFloat).run();
  EmbeddingTester(Device::getCuda(), DType::kQ4).run();
}

#endif  // LIBLLM_CUDA_ENABLED

}  // namespace libllm
