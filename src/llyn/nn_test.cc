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
#include "llyn/llyn.h"
#include "ly.h"
#include "lyutil/path.h"

using namespace llyn;

namespace lu = llyn::util;

constexpr int kDModel0 = 16;
constexpr int kDModel1 = 20;
constexpr int kSeqLen = 10;
constexpr int kBatchSize = 2;
constexpr int kNumHeads = 2;

namespace F = llyn::functional;

// test nn module that only have one input tensor and one return tensor.
template<class TModule>
void TestSingleInOutTensorModule(Context ctx,
                                 const std::string &model_path,
                                 const std::string &test_case_path,
                                 TModule *module) {
  readParameters(model_path, module);
  std::vector<Tensor> tensors = readAllTensors(test_case_path);

  CATCH_REQUIRE(tensors.size() % 2 == 0);
  for (int i = 0; i < tensors.size(); i += 2) {
    Tensor A = tensors[i];
    Tensor C_ref = tensors[i + 1];

    Tensor C = module->forward(A);
    CATCH_REQUIRE(F::allClose(C, C_ref));
  }
}

CATCH_TEST_CASE("test Linear module", "[core][nn][module]") {
  ly::Path model_dir = ly::Path("data") / "test";
  Context ctx = getCtxForCPU();

  ly::Path model_path = model_dir / "linear-model.params.bin";
  ly::Path tensor_file = model_dir / "linear-model.test_tensors.bin";

  auto linear = nn::Linear::create(ctx, kDModel0, kDModel1);
  TestSingleInOutTensorModule<nn::Linear>(
      ctx,
      model_path.string(),
      tensor_file.string(),
      linear.get());
}

CATCH_TEST_CASE("test LayerNorm module", "[core][nn][module]") {
  ly::Path model_dir = ly::Path("data") / "test";
  Context ctx = getCtxForCPU();

  ly::Path model_path = model_dir / "layer-norm-model.params.bin";
  ly::Path tensor_file = model_dir / "layer-norm-model.test_tensors.bin";

  auto layer_norm = nn::LayerNorm::create(ctx, kDModel0);
  TestSingleInOutTensorModule<nn::LayerNorm>(
      ctx,
      model_path.string(),
      tensor_file.string(),
      layer_norm.get());
}
