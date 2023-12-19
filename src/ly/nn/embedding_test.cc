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
#include "ly/internal/common.h"
#include "ly/operators/cpu/fingerprint.h"
#include "lyutil/random.h"
#include "lyutil/span.h"

namespace F = ly::functional;

namespace ly {
namespace nn {

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
    if (getWeightType() == DType::kQInt4Group32) {
      xr = {-0.8838, 0.6289, -0.8574, -0.2371, -0.5669, 0.7705, -1.0215, -0.1203};
    } else {
      xr = {-0.8596, 0.7052, -0.9062, -0.2155, -0.5044, 0.7694, -0.9440, -0.0745};
    }
    CATCH_REQUIRE(allClose(op::cpu::fingerprint(toCpu(x)), xr));
  }
};

CATCH_TEST_CASE("test nn::Embedding", "[ly][nn][embedding]") {
  EmbeddingTester(Device::getCpu(), DType::kFloat).run();
  EmbeddingTester(Device::getCpu(), DType::kQInt4Group32).run();
}

#ifdef LLYN_CUDA_ENABLED
CATCH_TEST_CASE("test nn::Embedding (cuda)", "[ly][nn][embedding][cuda]") {
  EmbeddingTester(Device::getCuda(), DType::kFloat).run();
  EmbeddingTester(Device::getCuda(), DType::kQInt4Group32).run();
}
#endif

}  // namespace nn
}  // namespace ly
