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

using llyn::Context;
using llyn::StateMap;
using llyn::Tensor;
using llyn::getCtxForCPU;
using llyn::nn::RMSNorm;

namespace F = llyn::functional;

CATCH_TEST_CASE("RMSNorm BVT", "[flint][module]") {
  Tensor x = Tensor::create<float>({8}, {
      0.3f, 0.4f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f, 0.8f});
  Tensor yRef = Tensor::create<float>({8}, {
      0.0612f, 0.1633f, 0.1225f, 0.2449f, 0.1633f, 0.3062f, 0.4287f, 0.6532f});

  Context ctx = getCtxForCPU();

  StateMap state_map;
  state_map.putTensor(RMSNorm::Weight, Tensor::create<float>({8}, {
      0.1f, 0.2f, 0.3f, 0.4f, 0.2f, 0.3f, 0.3f, 0.4f}));
  auto layer = RMSNorm::create(ctx, 8, 1e-5);
  layer->initParameters(state_map);

  Tensor y = layer->forward(x);
  CATCH_REQUIRE(F::allClose(y, yRef));
}
