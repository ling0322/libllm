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

#define POCKETFFT_NO_MULTITHREADING

#include <math.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "libllm/cpu/accessor.h"
#include "libllm/cpu/tensor.h"
#include "libllm/lut/thread_pool.h"
#include "libllm/mp.h"
#include "pocketfft/pocketfft_hdronly.h"

namespace libllm {
namespace op {
namespace cpu {

constexpr float PI = 3.1415926;
constexpr int NumFft = 400;
constexpr int NumPad = NumFft / 2;
constexpr int HopLength = 160;

constexpr int kMel_InputDim = 201;
constexpr int kMel_OutputDim = 80;
constexpr int kMel_Offsets[] = {
    1,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  14,
    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
    31,  32,  33,  35,  36,  37,  39,  40,  42,  44,  45,  47,  49,  51,  53,  55,
    57,  59,  61,  64,  66,  69,  71,  74,  77,  80,  83,  86,  90,  93,  97,  101,
    105, 109, 113, 117, 122, 127, 132, 137, 142, 148, 153, 159, 166, 172, 179, 186};
constexpr int kMel_Lengths[] = {
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2,
    2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 3, 4, 3,  3,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5, 5,
    6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 9, 10, 10, 10, 10, 11, 11, 11, 13, 13, 13, 14, 14, 14};
constexpr float kMel_Weights[] = {
    2.48626e-02f, 1.99082e-03f, 2.28718e-02f, 3.98164e-03f, 2.08810e-02f, 5.97247e-03f,
    1.88901e-02f, 7.96329e-03f, 1.68993e-02f, 9.95411e-03f, 1.49085e-02f, 1.19449e-02f,
    1.29177e-02f, 1.39358e-02f, 1.09268e-02f, 1.59266e-02f, 8.93602e-03f, 1.79174e-02f,
    6.94520e-03f, 1.99082e-02f, 4.95437e-03f, 2.18990e-02f, 2.96355e-03f, 2.38899e-02f,
    9.72731e-04f, 2.58807e-02f, 2.58353e-02f, 1.01809e-03f, 2.38445e-02f, 3.00891e-03f,
    2.18537e-02f, 4.99973e-03f, 1.98629e-02f, 6.99056e-03f, 1.78720e-02f, 8.98138e-03f,
    1.58812e-02f, 1.09722e-02f, 1.38904e-02f, 1.29630e-02f, 1.18996e-02f, 1.49538e-02f,
    9.90875e-03f, 1.69447e-02f, 7.91793e-03f, 1.89355e-02f, 5.92711e-03f, 2.08740e-02f,
    4.04043e-03f, 2.21142e-02f, 3.31861e-03f, 2.17367e-02f, 3.61097e-03f, 2.04977e-02f,
    4.76219e-03f, 1.84867e-02f, 6.59262e-03f, 1.58560e-02f, 8.96277e-03f, 1.27388e-02f,
    1.17513e-02f, 9.25037e-03f, 1.48531e-02f, 5.49084e-03f, 1.81775e-02f, 2.81555e-03f,
    1.54637e-03f, 1.63295e-02f, 7.42019e-03f, 1.11811e-02f, 1.20189e-02f, 6.06535e-03f,
    1.65613e-02f, 4.36088e-03f, 1.02980e-03f, 1.27705e-02f, 9.70719e-03f, 6.98640e-03f,
    1.48543e-02f, 4.39122e-03f, 1.41805e-03f, 1.14869e-02f, 1.00897e-02f, 4.00223e-04f,
    5.41110e-03f, 1.47356e-02f, 6.51819e-03f, 8.27841e-03f, 1.22776e-02f, 3.96781e-03f,
    2.18781e-03f, 1.01845e-02f, 9.98188e-03f, 2.28649e-03f, 3.86943e-03f, 1.12749e-02f,
    8.46622e-03f, 1.33977e-03f, 4.82029e-03f, 1.16783e-02f, 7.60868e-03f, 1.00910e-03f,
    5.15696e-03f, 1.15079e-02f, 7.30182e-03f, 1.19017e-03f, 4.98210e-03f, 1.08635e-02f,
    7.45119e-03f, 1.79138e-03f, 4.38592e-03f, 9.83249e-03f, 7.97396e-03f, 2.73259e-03f,
    3.44745e-03f, 8.49135e-03f, 8.79769e-03f, 3.94383e-03f, 2.23576e-03f, 6.90675e-03f,
    9.85924e-03f, 5.36424e-03f, 8.69234e-04f, 8.11000e-04f, 5.13665e-03f, 9.46230e-03f,
    6.94108e-03f, 2.77840e-03f, 3.23120e-03f, 7.23705e-03f, 8.62883e-03f, 4.77391e-03f,
    9.18991e-04f, 1.23364e-03f, 4.94332e-03f, 8.65301e-03f, 6.81850e-03f, 3.24858e-03f,
    2.61644e-03f, 6.05185e-03f, 8.88047e-03f, 5.57448e-03f, 2.26849e-03f, 2.86367e-04f,
    3.46780e-03f, 6.64923e-03f, 7.87146e-03f, 4.80990e-03f, 1.74833e-03f, 9.24591e-04f,
    3.87081e-03f, 6.81703e-03f, 7.28334e-03f, 4.44812e-03f, 1.61290e-03f, 1.17033e-03f,
    3.89873e-03f, 6.62713e-03f, 7.04732e-03f, 4.42171e-03f, 1.79611e-03f, 1.08930e-03f,
    3.61598e-03f, 6.14267e-03f, 7.10294e-03f, 4.67145e-03f, 2.23996e-03f, 7.39228e-04f,
    3.07911e-03f, 5.41899e-03f, 7.39719e-03f, 5.14546e-03f, 2.89374e-03f, 6.42016e-04f,
    1.70687e-04f, 2.33757e-03f, 4.50446e-03f, 6.67135e-03f, 5.79848e-03f, 3.71323e-03f,
    1.62798e-03f, 1.43453e-03f, 3.44122e-03f, 5.44790e-03f, 6.59109e-03f, 4.66001e-03f,
    2.72893e-03f, 7.97849e-04f, 4.07504e-04f, 2.26583e-03f, 4.12416e-03f, 5.98248e-03f,
    5.70082e-03f, 3.91251e-03f, 2.12420e-03f, 3.35886e-04f, 1.00991e-03f, 2.73085e-03f,
    4.45178e-03f, 6.17272e-03f, 5.15091e-03f, 3.49481e-03f, 1.83871e-03f, 1.82611e-04f,
    1.29437e-03f, 2.88807e-03f, 4.48178e-03f, 6.07548e-03f, 4.88666e-03f, 3.35300e-03f,
    1.81934e-03f, 2.85683e-04f, 1.31314e-03f, 2.78902e-03f, 4.26489e-03f, 5.74077e-03f,
    4.85998e-03f, 3.43971e-03f, 2.01943e-03f, 5.99162e-04f, 1.11217e-03f, 2.47893e-03f,
    3.84569e-03f, 5.21246e-03f, 5.02864e-03f, 3.71337e-03f, 2.39810e-03f, 1.08283e-03f,
    7.31755e-04f, 1.99747e-03f, 3.26318e-03f, 4.52890e-03f, 5.35569e-03f, 4.13766e-03f,
    2.91963e-03f, 1.70160e-03f, 4.83576e-04f, 2.07140e-04f, 1.37928e-03f, 2.55141e-03f,
    3.72355e-03f, 4.89569e-03f, 4.68090e-03f, 3.55292e-03f, 2.42494e-03f, 1.29697e-03f,
    1.68990e-04f, 6.54527e-04f, 1.74001e-03f, 2.82548e-03f, 3.91096e-03f, 4.99644e-03f,
    4.27098e-03f, 3.22640e-03f, 2.18181e-03f, 1.13723e-03f, 9.26491e-05f, 8.54627e-04f,
    1.85985e-03f, 2.86508e-03f, 3.87031e-03f, 4.87553e-03f, 4.08314e-03f, 3.11578e-03f,
    2.14843e-03f, 1.18108e-03f, 2.13721e-04f, 8.48342e-04f, 1.77925e-03f, 2.71016e-03f,
    3.64107e-03f, 4.57197e-03f, 4.07973e-03f, 3.18389e-03f, 2.28806e-03f, 1.39222e-03f,
    4.96387e-04f, 6.71620e-04f, 1.53370e-03f, 2.39579e-03f, 3.25787e-03f, 4.11996e-03f,
    4.22773e-03f, 3.39812e-03f, 2.56852e-03f, 1.73891e-03f, 9.09308e-04f, 7.97036e-05f,
    3.55980e-04f, 1.15433e-03f, 1.95268e-03f, 2.75102e-03f, 3.54937e-03f, 4.34772e-03f,
    3.72996e-03f, 2.96169e-03f, 2.19342e-03f, 1.42515e-03f, 6.56883e-04f, 6.68295e-04f,
    1.40762e-03f, 2.14694e-03f, 2.88627e-03f, 3.62559e-03f, 4.15458e-03f, 3.44311e-03f,
    2.73164e-03f, 2.02017e-03f, 1.30870e-03f, 5.97227e-04f, 9.92651e-05f, 7.83930e-04f,
    1.46859e-03f, 2.15326e-03f, 2.83792e-03f, 3.52259e-03f, 3.99152e-03f, 3.33265e-03f,
    2.67378e-03f, 2.01491e-03f, 1.35604e-03f, 6.97171e-04f, 3.83011e-05f, 1.01811e-04f,
    7.35857e-04f, 1.36990e-03f, 2.00395e-03f, 2.63799e-03f, 3.27204e-03f, 3.90609e-03f,
    3.36826e-03f, 2.75810e-03f, 2.14794e-03f, 1.53778e-03f, 9.27626e-04f, 3.17468e-04f,
    5.53036e-04f, 1.14021e-03f, 1.72738e-03f, 2.31454e-03f, 2.90171e-03f, 3.48888e-03f,
    3.52334e-03f, 2.95829e-03f, 2.39325e-03f, 1.82820e-03f, 1.26315e-03f, 6.98103e-04f,
    1.33056e-04f, 2.60839e-04f, 8.04597e-04f, 1.34836e-03f, 1.89211e-03f, 2.43587e-03f,
    2.97963e-03f, 3.52339e-03f, 3.25138e-03f, 2.72811e-03f, 2.20484e-03f, 1.68156e-03f,
    1.15829e-03f, 6.35020e-04f, 1.11747e-04f, 3.84981e-04f, 8.88539e-04f, 1.39210e-03f,
    1.89565e-03f, 2.39921e-03f, 2.90277e-03f, 3.40633e-03f, 3.13276e-03f, 2.64818e-03f,
    2.16359e-03f, 1.67901e-03f, 1.19442e-03f, 7.09836e-04f, 2.25250e-04f, 3.66742e-04f,
    8.33070e-04f, 1.29940e-03f, 1.76573e-03f, 2.23205e-03f, 2.69838e-03f, 3.16471e-03f,
    3.14131e-03f, 2.69255e-03f, 2.24380e-03f, 1.79504e-03f, 1.34628e-03f, 8.97518e-04f,
    4.48759e-04f};

std::vector<float> hannWindow(int windowSize) {
  std::vector<float> window(windowSize);
  for (int i = 0; i < windowSize; ++i) {
    window[i] = static_cast<float>(0.5 - cosf(static_cast<float>(2 * PI * i / windowSize)) / 2);
  }

  return window;
}

std::vector<std::complex<float>> fft(lut::Span<const float> input) {
  CHECK(input.size() % 2 == 0);

  std::vector<std::complex<float>> output(input.size() / 2 + 1);

  pocketfft::shape_t shapeIn = {input.size()};
  pocketfft::stride_t strideIn = {sizeof(float)};
  pocketfft::stride_t strideOut = {sizeof(std::complex<float>)};
  pocketfft::r2c(
      shapeIn,
      strideIn,
      strideOut,
      0,
      pocketfft::FORWARD,
      input.data(),
      output.data(),
      1.0f);

  return output;
}

std::vector<float> mel80FilterBank(lut::Span<const float> input) {
  CHECK(input.size() == kMel_InputDim);

  std::vector<float> output(kMel_OutputDim);

  const float *weight = kMel_Weights;
  for (int mel_bin = 0; mel_bin < kMel_OutputDim; ++mel_bin) {
    float v = 0;

    int begin = kMel_Offsets[mel_bin];
    int end = kMel_Offsets[mel_bin] + kMel_Lengths[mel_bin];
    for (int i = begin; i < end; ++i) {
      v += input[i] * (*weight);
      ++weight;
    }

    output[mel_bin] = v;
  }

  CHECK(weight - kMel_Weights == sizeof(kMel_Weights) / sizeof(float));
  return output;
}

std::vector<float> applyLogMelSpectrogramWindow(
    lut::Span<const float> data,
    lut::Span<const float> window) {
  CHECK(data.size() == window.size());

  // apply window.
  std::vector<float> windowData(data.size());
  for (int i = 0; i < data.size(); ++i) {
    windowData[i] = data[i] * window[i];
  }

  // apply fft.
  std::vector<std::complex<float>> fftResult = fft(windowData);

  // compute magnitudes.
  std::vector<float> magnitudes(fftResult.size());
  for (int i = 0; i < fftResult.size(); ++i) {
    float v = std::abs(fftResult[i]);
    magnitudes[i] = v * v;
  }

  // apply mel filter-bank.
  std::vector<float> melFbank = mel80FilterBank(magnitudes);

  // apply log10 to mel filter-bank.
  for (int i = 0; i < melFbank.size(); ++i) {
    float v = std::max(1e-10f, melFbank[i]);
    melFbank[i] = log10f(v);
  }

  return melFbank;
}

Tensor logMelSpectrogram(Tensor inputs) {
  CHECK(inputs.getDim() == 1);
  CHECK(inputs.getShape(0) > NumPad);
  LOG(INFO) << inputs.getShape(0);

  int numFrames = (inputs.getShape(0) + NumPad - (NumFft - HopLength)) / HopLength;
  LOG(INFO) << numFrames;
  Tensor outputs = op::cpu::tensor({numFrames, kMel_OutputDim}, DType::kFloat);

  TensorAccessor<const float, 1> inputAccessor(inputs);
  lut::Span<const float> inputSpan(inputAccessor.getData(), inputAccessor.getShape(0));

  // padding.
  std::vector<float> paddedInputs(NumPad);
  std::copy(inputSpan.begin() + 1, inputSpan.begin() + 1 + NumPad, paddedInputs.rbegin());
  paddedInputs.insert(paddedInputs.end(), inputSpan.begin(), inputSpan.end());
  paddedInputs.insert(paddedInputs.end(), inputSpan.rbegin() + 1, inputSpan.rbegin() + 1 + NumPad);
  CHECK(paddedInputs.size() == inputSpan.size() + 2 * NumPad);

  // hanning window function.
  std::vector<float> window = hannWindow(NumFft);

  // for each window.
  TensorAccessor<float, 2> outputAccessor(outputs);
  lut::Span<const float> paddedSpan(paddedInputs);
  for (int i = 0; i < numFrames; ++i) {
    lut::Span<const float> windowSpan = paddedSpan.subspan(i * HopLength, NumFft);
    std::vector<float> feature = applyLogMelSpectrogramWindow(windowSpan, window);
    CHECK(feature.size() == outputAccessor.getShape(1));

    for (int j = 0; j < feature.size(); ++j) {
      outputAccessor[i][j] = feature[j];
    }
  }

  // whisper feature normalize.
  float maxVal = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < numFrames; ++i) {
    for (int j = 0; j < kMel_OutputDim; ++j) {
      float val = outputAccessor[i][j];
      if (val > maxVal) maxVal = val;
    }
  }

  float featureMinVal = maxVal - 8.0f;
  for (int i = 0; i < numFrames; ++i) {
    for (int j = 0; j < kMel_OutputDim; ++j) {
      float val = outputAccessor[i][j];
      val = std::max(val, featureMinVal);
      outputAccessor[i][j] = (val + 4.0f) / 4.0f;
    }
  }

  return outputs;
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
