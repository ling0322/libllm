// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "libllm/sampling_batch.h"

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
#include "flint/operators.h"

namespace libllm {

CATCH_TEST_CASE("SamplingBatch validates metadata", "[libllm][sampling_batch]") {
  SamplingBatch empty({}, {}, {}, {});
  CATCH_REQUIRE(empty.empty());
  CATCH_REQUIRE(empty.size() == 0);
  CATCH_REQUIRE(empty.sequenceIndices().empty());
  empty.prepare(fl::Device::getCpu());
  empty.prepare(fl::Device::getCpu());
}

CATCH_TEST_CASE("SamplingBatch gathers and samples selected rows", "[libllm][sampling_batch]") {
  fl::Tensor cpuLogits = fl::Tensor::create<float>(
      {4, 4},
      {0.0f, 5.0f, 1.0f, 2.0f,
       7.0f, 1.0f, 0.0f, 2.0f,
       1.0f, 2.0f, 8.0f, 0.0f,
       0.0f, 1.0f, 2.0f, 9.0f});

  std::vector<fl::Device> devices{fl::Device::getCpu()};
  if (fl::isOperatorsAvailable(fl::Device::kCuda)) {
    devices.push_back(fl::Device::getCuda());
  }

  for (const fl::Device &device : devices) {
    CATCH_INFO("device = " << device.getName());
    SamplingBatch batch({2, 0, 3}, {0.0f, 0.0f, 0.0f}, {0, 1, 0}, {1.0f, 1.0f, 0.1f});
    CATCH_REQUIRE_FALSE(batch.empty());
    CATCH_REQUIRE(batch.size() == 3);
    CATCH_REQUIRE(batch.sequenceIndices() == std::vector<fl::LongType>{2, 0, 3});

    batch.prepare(device);
    batch.prepare(device);
    fl::Tensor sampled = batch.sample(fl::F::to(device, cpuLogits));
    sampled = fl::F::to(fl::Device::getCpu(), sampled);

    CATCH_REQUIRE(sampled.getShape() == std::vector<int>{3});
    const fl::LongType *data = sampled.getInternalData()->getData<fl::LongType>(
        sampled.getInternalOffset());
    CATCH_REQUIRE(data[0] == 2);
    CATCH_REQUIRE(data[1] == 1);
    CATCH_REQUIRE(data[2] == 3);
  }
}

}  // namespace libllm
