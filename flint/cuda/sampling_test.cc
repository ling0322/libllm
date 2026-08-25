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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {

namespace {

std::vector<LongType> sampleOnCuda(
    int rows,
    int vocabSize,
    const std::vector<float> &logits,
    const std::vector<float> &temperatures,
    const std::vector<IntType> &topKs,
    const std::vector<float> &topPs,
    DType logitsType = DType::kFloat) {
  Tensor cudaLogits = F::to(Device::getCuda(), Tensor::create<float>({rows, vocabSize}, logits));
  if (logitsType != DType::kFloat) cudaLogits = F::cast(cudaLogits, logitsType);

  Tensor sampled = F::sample(
      cudaLogits,
      F::to(Device::getCuda(), Tensor::create<float>({rows}, temperatures)),
      F::to(Device::getCuda(), Tensor::create<IntType>({rows}, topKs)),
      F::to(Device::getCuda(), Tensor::create<float>({rows}, topPs)));
  sampled = F::to(Device::getCpu(), sampled);
  const LongType *data = sampled.getInternalData()->getData<LongType>(sampled.getInternalOffset());
  return std::vector<LongType>(data, data + rows);
}

void checkEmpiricalDistribution(
    const std::vector<float> &rowLogits,
    float temperature,
    IntType topK,
    float topP,
    const std::vector<float> &expected,
    uint64_t seed) {
  constexpr int NumSamples = 32768;
  int vocabSize = static_cast<int>(rowLogits.size());
  std::vector<float> logits;
  logits.reserve(static_cast<size_t>(NumSamples) * vocabSize);
  for (int sample = 0; sample < NumSamples; ++sample) {
    logits.insert(logits.end(), rowLogits.begin(), rowLogits.end());
  }

  F::manualSeed(Device::getCuda(), seed);
  std::vector<LongType> sampled = sampleOnCuda(
      NumSamples,
      vocabSize,
      logits,
      std::vector<float>(NumSamples, temperature),
      std::vector<IntType>(NumSamples, topK),
      std::vector<float>(NumSamples, topP));

  CATCH_REQUIRE(std::all_of(sampled.begin(), sampled.end(), [&](LongType label) {
    return label >= 0 && label < vocabSize;
  }));
  std::vector<int> counts(vocabSize, 0);
  for (LongType label : sampled) ++counts[label];

  for (int label = 0; label < vocabSize; ++label) {
    float observed = static_cast<float>(counts[label]) / NumSamples;
    float probability = expected[label];
    CATCH_INFO("label=" << label << " observed=" << observed << " expected=" << probability);
    if (probability == 0.0f) {
      CATCH_REQUIRE(counts[label] == 0);
    } else {
      float standardError = std::sqrt(probability * (1.0f - probability) / NumSamples);
      CATCH_REQUIRE(std::abs(observed - probability) <= 6.0f * standardError + 1.0e-3f);
    }
  }
}

}  // namespace

CATCH_TEST_CASE("test CUDA batched sampling parameters", "[fl][op][cuda][sampling]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor logits = Tensor::create<float>(
      {4, 4},
      {std::numeric_limits<float>::quiet_NaN(),
       4.0f,
       2.0f,
       1.0f,
       0.0f,
       1.0f,
       5.0f,
       2.0f,
       0.0f,
       1.0f,
       2.0f,
       6.0f,
       4.0f,
       3.0f,
       0.0f,
       -1.0f});
  Tensor temperatures = Tensor::create<float>({4}, {0.0f, 1.0f, 1.0f, 0.7f});
  Tensor topKs = Tensor::create<IntType>({4}, {0, 1, 0, 2});
  Tensor topPs = Tensor::create<float>({4}, {1.0f, 1.0f, 0.1f, 0.9f});

  logits = F::to(Device::getCuda(), logits);
  temperatures = F::to(Device::getCuda(), temperatures);
  topKs = F::to(Device::getCuda(), topKs);
  topPs = F::to(Device::getCuda(), topPs);

  F::manualSeed(Device::getCuda(), 1234);
  Tensor first = F::sample(logits, temperatures, topKs, topPs);
  F::manualSeed(Device::getCuda(), 1234);
  Tensor second = F::sample(logits, temperatures, topKs, topPs);
  first = F::to(Device::getCpu(), first);
  second = F::to(Device::getCpu(), second);

  CATCH_REQUIRE(first.getShape() == std::vector<int>{4});
  const LongType *firstData = first.getInternalData()->getData<LongType>(first.getInternalOffset());
  const LongType *secondData = second.getInternalData()->getData<LongType>(
      second.getInternalOffset());
  CATCH_REQUIRE(firstData[0] == 1);
  CATCH_REQUIRE(firstData[1] == 2);
  CATCH_REQUIRE(firstData[2] == 3);
  CATCH_REQUIRE((firstData[3] == 0 || firstData[3] == 1));
  for (int row = 0; row < 4; ++row) CATCH_REQUIRE(firstData[row] == secondData[row]);

  constexpr int vocabSize = 128256;
  std::vector<float> largeLogits(vocabSize, 0.0f);
  largeLogits[123456] = 10.0f;
  Tensor largeLogitsCuda = F::cast(
      F::to(Device::getCuda(), Tensor::create<float>({1, vocabSize}, largeLogits)),
      DType::kFloat16);
  Tensor largeSample = F::sample(
      largeLogitsCuda,
      F::to(Device::getCuda(), Tensor::create<float>({1}, {1.0f})),
      F::to(Device::getCuda(), Tensor::create<IntType>({1}, {2048})),
      F::to(Device::getCuda(), Tensor::create<float>({1}, {0.8f})));
  largeSample = F::to(Device::getCpu(), largeSample);
  LongType largeToken = largeSample.getInternalData()->getData<LongType>(
      largeSample.getInternalOffset())[0];
  CATCH_REQUIRE(largeToken == 123456);
}

CATCH_TEST_CASE("test CUDA sampling threshold ties and top-p", "[fl][op][cuda][sampling]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int rows = 32;
  constexpr int vocabSize = 600;
  std::vector<float> tiedLogits(rows * vocabSize, 0.0f);
  std::vector<float> temperatures(rows, 1.0f);
  std::vector<IntType> topKs(rows, 257);
  std::vector<float> topPs(rows, 1.0f);
  Tensor tiedSamples = F::sample(
      F::to(Device::getCuda(), Tensor::create<float>({rows, vocabSize}, tiedLogits)),
      F::to(Device::getCuda(), Tensor::create<float>({rows}, temperatures)),
      F::to(Device::getCuda(), Tensor::create<IntType>({rows}, topKs)),
      F::to(Device::getCuda(), Tensor::create<float>({rows}, topPs)));
  tiedSamples = F::to(Device::getCpu(), tiedSamples);
  const LongType *tiedData = tiedSamples.getInternalData()->getData<LongType>(
      tiedSamples.getInternalOffset());
  for (int row = 0; row < rows; ++row) CATCH_REQUIRE(tiedData[row] < 257);

  Tensor truncatedSample = F::sample(
      F::to(Device::getCuda(), Tensor::create<float>({1, 3}, {5.0f, 4.0f, 3.0f})),
      F::to(Device::getCuda(), Tensor::create<float>({1}, {1.0f})),
      F::to(Device::getCuda(), Tensor::create<IntType>({1}, {3})),
      F::to(Device::getCuda(), Tensor::create<float>({1}, {0.5f})));
  truncatedSample = F::to(Device::getCpu(), truncatedSample);
  CATCH_REQUIRE(
      truncatedSample.getInternalData()->getData<LongType>(
          truncatedSample.getInternalOffset())[0] == 0);
}

CATCH_TEST_CASE(
    "test CUDA sampling handles 131072-token vocabularies",
    "[fl][op][cuda][sampling][large-vocab]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int rows = 4;
  constexpr int vocabSize = 131072;
  std::vector<float> logits(rows * vocabSize, -100.0f);

  logits[0] = 10.0f;
  logits[vocabSize + vocabSize - 1] = 10.0f;
  std::fill(logits.begin() + 2 * vocabSize, logits.begin() + 3 * vocabSize, 0.0f);
  logits[3 * vocabSize + 65537] = 10.0f;

  std::vector<LongType> sampled = sampleOnCuda(
      rows,
      vocabSize,
      logits,
      {1.0f, 1.0f, 1.0f, 1.0f},
      {2048, 2048, 2048, 0},
      {0.5f, 0.5f, 1.0f, 0.5f});

  CATCH_REQUIRE(sampled[0] == 0);
  CATCH_REQUIRE(sampled[1] == vocabSize - 1);
  CATCH_REQUIRE(sampled[2] >= 0);
  CATCH_REQUIRE(sampled[2] < 2048);
  CATCH_REQUIRE(sampled[3] == 65537);
}

CATCH_TEST_CASE(
    "test CUDA sampling orders special floating-point logits",
    "[fl][op][cuda][sampling]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  float infinity = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> logits = {nan,       -infinity, -3.0f,     -1.0f,     -2.0f,     infinity,
                               nan,       nan,       nan,       nan,       nan,       nan,
                               -3.0f,     -1.0f,     -2.0f,     -4.0f,     -5.0f,     -6.0f,
                               infinity,  infinity,  1.0f,      0.0f,      -1.0f,     -infinity,
                               -infinity, -infinity, -infinity, -infinity, -infinity, -infinity};
  std::vector<LongType> sampled = sampleOnCuda(
      5,
      6,
      logits,
      {0.0f, 0.0f, 0.0f, 1.0f, 1.0f},
      {0, 1, 6, 6, 3},
      {1.0f, 1.0f, 1.0f, 0.5f, 1.0f});

  CATCH_REQUIRE(sampled[0] == 5);
  CATCH_REQUIRE(sampled[1] == 0);
  CATCH_REQUIRE(sampled[2] == 1);
  CATCH_REQUIRE((sampled[3] == 0 || sampled[3] == 1));
  CATCH_REQUIRE(sampled[4] < 3);
}

CATCH_TEST_CASE(
    "test CUDA sampling covers top-k execution boundaries",
    "[fl][op][cuda][sampling]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int rows = 5;
  constexpr int vocabSize = 2051;
  std::vector<float> logits(rows * vocabSize, 0.0f);
  for (int row = 0; row < rows; ++row) logits[row * vocabSize + 2050] = 10.0f;
  std::vector<LongType> boundarySamples = sampleOnCuda(
      rows,
      vocabSize,
      logits,
      std::vector<float>(rows, 1.0f),
      {255, 256, 257, 2048, 2049},
      std::vector<float>(rows, 0.8f));
  for (LongType label : boundarySamples) CATCH_REQUIRE(label == 2050);

  std::vector<float> oneRowLogits(vocabSize);
  for (int label = 0; label < vocabSize; ++label) {
    oneRowLogits[label] = static_cast<float>((label * 37) % 101) / 10.0f;
  }
  auto sampleWithTopK = [&](IntType topK) {
    F::manualSeed(Device::getCuda(), 9876);
    return sampleOnCuda(1, vocabSize, oneRowLogits, {1.0f}, {topK}, {1.0f})[0];
  };
  LongType disabledWithMinusOne = sampleWithTopK(-1);
  CATCH_REQUIRE(sampleWithTopK(0) == disabledWithMinusOne);
  CATCH_REQUIRE(sampleWithTopK(vocabSize) == disabledWithMinusOne);

  std::vector<LongType> topOne = sampleOnCuda(
      1,
      vocabSize,
      oneRowLogits,
      {std::numeric_limits<float>::max()},
      {1},
      {1.0f});
  CATCH_REQUIRE(oneRowLogits[topOne[0]] == 10.0f);
}

CATCH_TEST_CASE(
    "test CUDA sampling handles temperature and vocabulary limits",
    "[fl][op][cuda][sampling]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::vector<float> temperatures = {
      std::numeric_limits<float>::denorm_min(),
      1.0e-6f,
      1.0f,
      std::numeric_limits<float>::max()};
  std::vector<float> logits;
  for (int row = 0; row < 4; ++row) {
    logits.insert(logits.end(), {4.0f, 3.0f, 2.0f, 1.0f});
  }
  for (DType dtype : {DType::kFloat, DType::kFloat16}) {
    std::vector<LongType> sampled = sampleOnCuda(
        4,
        4,
        logits,
        temperatures,
        {4, 4, 4, 4},
        {0.24f, 0.24f, 0.24f, 0.24f},
        dtype);
    for (LongType label : sampled) CATCH_REQUIRE(label == 0);
  }

  std::vector<LongType> singletonSamples = sampleOnCuda(
      3,
      1,
      {3.0f, std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity()},
      {0.0f, 1.0f, std::numeric_limits<float>::max()},
      {-1, 0, 1},
      {std::numeric_limits<float>::denorm_min(), 1.0f, 0.5f});
  CATCH_REQUIRE(singletonSamples == std::vector<LongType>{0, 0, 0});
}

CATCH_TEST_CASE(
    "test CUDA sampling matches categorical distributions",
    "[fl][op][cuda][sampling][statistical]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  std::vector<float> logits = {std::log(0.4f), std::log(0.3f), std::log(0.2f), std::log(0.1f)};

  CATCH_SECTION("softmax distribution") {
    checkEmpiricalDistribution(logits, 1.0f, 0, 1.0f, {0.4f, 0.3f, 0.2f, 0.1f}, 1001);
  }
  CATCH_SECTION("top-k distribution is renormalized") {
    checkEmpiricalDistribution(logits, 1.0f, 2, 1.0f, {4.0f / 7.0f, 3.0f / 7.0f, 0.0f, 0.0f}, 1002);
  }
  CATCH_SECTION("top-p distribution is truncated and renormalized") {
    checkEmpiricalDistribution(
        logits,
        1.0f,
        0,
        0.85f,
        {4.0f / 9.0f, 3.0f / 9.0f, 2.0f / 9.0f, 0.0f},
        1003);
  }
  CATCH_SECTION("temperature changes the distribution") {
    checkEmpiricalDistribution(
        logits,
        0.5f,
        0,
        1.0f,
        {16.0f / 30.0f, 9.0f / 30.0f, 4.0f / 30.0f, 1.0f / 30.0f},
        1004);
  }
}

}  // namespace fl