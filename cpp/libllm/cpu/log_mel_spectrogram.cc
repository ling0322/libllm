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
#include "libllm/mp.h"
#include "lutil/thread_pool.h"
#include "pocketfft/pocketfft_hdronly.h"

namespace libllm {
namespace op {
namespace cpu {

constexpr float PI = 3.1415926;
constexpr int NumFft = 400;
constexpr int NumPad = NumFft / 2;
constexpr int HopLength = 160;

constexpr int kMel_InputDim = 201;
constexpr int kMel_OutputDim = 128;
constexpr int kMel_Offsets[] = {
    1,   1,   2,   2,   3,   3,   4,   5,   5,   6,   6,   7,   8,   8,   9,   9,   10,  10,  11,
    12,  12,  13,  13,  14,  15,  15,  16,  16,  17,  17,  18,  19,  19,  20,  20,  21,  22,  22,
    23,  23,  24,  24,  25,  26,  26,  27,  28,  28,  29,  30,  30,  31,  32,  32,  33,  34,  35,
    36,  37,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  48,  49,  50,  51,  52,  54,  55,
    56,  58,  59,  60,  62,  63,  65,  66,  68,  70,  71,  73,  75,  77,  79,  80,  82,  84,  86,
    89,  91,  93,  95,  98,  100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 136,
    140, 143, 147, 150, 154, 158, 161, 165, 169, 174, 178, 182, 187, 191};
constexpr int kMel_Lengths[] = {
    1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1,
    1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2,
    2, 2, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 4, 3, 3, 4, 4, 4, 3, 3, 4, 4, 5, 5, 4,
    4, 5, 5, 4, 5, 5, 5, 6, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 7, 7, 8, 9, 9, 8, 9, 9, 9, 9};
constexpr float kMel_Weights[] = {
    1.23740e-02f, 3.03926e-02f, 2.47480e-02f, 1.80186e-02f, 3.71220e-02f, 5.64459e-03f,
    6.72939e-03f, 3.60372e-02f, 1.91034e-02f, 2.36632e-02f, 3.14774e-02f, 1.12892e-02f,
    1.08480e-03f, 4.16818e-02f, 1.34588e-02f, 2.93078e-02f, 2.58328e-02f, 1.69338e-02f,
    3.82068e-02f, 4.55979e-03f, 7.81420e-03f, 3.49524e-02f, 2.01882e-02f, 2.25784e-02f,
    3.25622e-02f, 1.02044e-02f, 2.16960e-03f, 4.05969e-02f, 1.45436e-02f, 2.82230e-02f,
    2.69176e-02f, 1.58490e-02f, 3.92916e-02f, 3.47499e-03f, 8.89900e-03f, 3.38676e-02f,
    2.12730e-02f, 2.14936e-02f, 3.36470e-02f, 9.11958e-03f, 3.25441e-03f, 3.95121e-02f,
    1.56284e-02f, 2.71382e-02f, 2.80024e-02f, 1.47642e-02f, 4.03764e-02f, 2.38069e-03f,
    1.02026e-02f, 3.16115e-02f, 2.45470e-02f, 1.53292e-02f, 1.66584e-03f, 3.67291e-02f,
    2.00971e-02f, 1.69310e-02f, 2.90266e-03f, 3.28450e-02f, 2.35200e-02f, 1.10389e-02f,
    1.07258e-02f, 2.27183e-02f, 3.22787e-02f, 1.16268e-04f, 2.28535e-02f, 8.56344e-03f,
    1.49798e-02f, 1.55140e-02f, 8.51491e-03f, 2.11068e-02f, 3.32652e-03f, 2.54706e-02f,
    2.73591e-02f, 6.58536e-04f, 2.38381e-02f, 3.44359e-03f, 2.12246e-02f, 5.35842e-03f,
    1.94256e-02f, 6.49325e-03f, 1.83554e-02f, 6.93138e-03f, 1.79350e-02f, 6.74968e-03f,
    1.80915e-02f, 6.01899e-03f, 1.87577e-02f, 4.80453e-03f, 1.98717e-02f, 3.16628e-03f,
    2.13769e-02f, 1.25317e-03f, 1.15934e-03f, 2.08036e-02f, 4.04487e-03f, 1.75536e-02f,
    7.08320e-03f, 1.40754e-02f, 1.03266e-02f, 1.04092e-02f, 1.37370e-02f, 6.59188e-03f,
    1.72799e-02f, 1.46804e-03f, 2.65682e-03f, 1.80919e-02f, 5.85656e-03f, 1.33428e-02f,
    1.02827e-02f, 8.56800e-03f, 1.47223e-02f, 1.04040e-03f, 3.79086e-03f, 1.71468e-02f,
    6.11609e-03f, 1.17593e-02f, 1.11339e-02f, 6.43858e-03f, 1.60781e-02f, 4.23917e-03f,
    1.19989e-03f, 1.27567e-02f, 9.65299e-03f, 7.06935e-03f, 1.49405e-02f, 4.19025e-03f,
    1.51483e-03f, 1.20090e-02f, 9.84823e-03f, 6.10224e-03f, 1.53386e-02f, 5.57677e-03f,
    3.68273e-04f, 9.89749e-03f, 1.13534e-02f, 2.05122e-03f, 3.89297e-03f, 1.29735e-02f,
    8.06632e-03f, 6.74493e-03f, 1.38587e-02f, 5.41191e-03f, 7.42202e-04f, 8.98779e-03f,
    1.13787e-02f, 3.32958e-03f, 2.82314e-03f, 1.06805e-02f, 9.43341e-03f, 1.76326e-03f,
    4.39019e-03f, 1.18776e-02f, 7.97006e-03f, 6.61047e-04f, 5.49467e-03f, 1.26295e-02f,
    6.93988e-03f, 6.18402e-03f, 1.29347e-02f, 6.29779e-03f, 2.32521e-05f, 6.50207e-03f,
    1.23266e-02f, 6.00217e-03f, 3.15488e-04f, 6.48926e-03f, 1.20413e-02f, 6.01463e-03f,
    2.99796e-04f, 6.18288e-03f, 1.20427e-02f, 6.29981e-03f, 5.56896e-04f, 1.12047e-05f,
    5.61729e-03f, 1.12234e-02f, 6.82516e-03f, 1.35264e-03f, 4.82410e-03f, 1.01662e-02f,
    7.56076e-03f, 2.34590e-03f, 3.83236e-03f, 8.92296e-03f, 8.47910e-03f, 3.50979e-03f,
    2.66873e-03f, 7.51965e-03f, 9.55501e-03f, 4.81966e-03f, 8.43175e-05f, 1.35767e-03f,
    5.98020e-03f, 1.06027e-02f, 6.25298e-03f, 1.74060e-03f, 4.32644e-03f, 8.73132e-03f,
    7.78917e-03f, 3.48924e-03f, 2.57835e-03f, 6.77583e-03f, 9.40942e-03f, 5.31195e-03f,
    1.21448e-03f, 7.54112e-04f, 4.75396e-03f, 8.75380e-03f, 7.19209e-03f, 3.28754e-03f,
    2.68180e-03f, 6.49331e-03f, 9.11458e-03f, 5.39387e-03f, 1.67317e-03f, 5.73943e-04f,
    4.20600e-03f, 7.83806e-03f, 7.52023e-03f, 3.97471e-03f, 4.29187e-04f, 1.90464e-03f,
    5.36569e-03f, 8.82674e-03f, 6.27609e-03f, 2.89751e-03f, 2.89885e-03f, 6.19694e-03f,
    8.56699e-03f, 5.34748e-03f, 2.12797e-03f, 4.47502e-04f, 3.59030e-03f, 6.73311e-03f,
    7.77024e-03f, 4.70231e-03f, 1.63439e-03f, 1.01536e-03f, 4.01019e-03f, 7.00501e-03f,
    7.23443e-03f, 4.31096e-03f, 1.38748e-03f, 1.33349e-03f, 4.18731e-03f, 7.04113e-03f,
    6.93188e-03f, 4.14606e-03f, 1.36023e-03f, 1.42880e-03f, 4.14825e-03f, 6.86770e-03f,
    6.83705e-03f, 4.18239e-03f, 1.52774e-03f, 1.32610e-03f, 3.91751e-03f, 6.50892e-03f,
    6.92640e-03f, 4.39673e-03f, 1.86706e-03f, 1.04828e-03f, 3.51767e-03f, 5.98707e-03f,
    7.17824e-03f, 4.76768e-03f, 2.35712e-03f, 6.16364e-04f, 2.96949e-03f, 5.32262e-03f,
    7.57265e-03f, 5.27559e-03f, 2.97852e-03f, 6.81461e-04f, 4.97140e-05f, 2.29205e-03f,
    4.53438e-03f, 6.77672e-03f, 5.90241e-03f, 3.71350e-03f, 1.52459e-03f, 1.50285e-03f,
    3.63961e-03f, 5.77637e-03f, 6.63159e-03f, 4.54574e-03f, 2.45990e-03f, 3.74049e-04f,
    6.17959e-04f, 2.65411e-03f, 4.69026e-03f, 6.72641e-03f, 5.46035e-03f, 3.47271e-03f,
    1.48507e-03f, 1.59234e-03f, 3.53262e-03f, 5.47290e-03f, 6.44368e-03f, 4.54963e-03f,
    2.65558e-03f, 7.61525e-04f, 4.67494e-04f, 2.31642e-03f, 4.16534e-03f, 6.01427e-03f,
    5.67845e-03f, 3.87357e-03f, 2.06870e-03f, 2.63827e-04f, 1.05349e-03f, 2.81536e-03f,
    4.57723e-03f, 6.33910e-03f, 5.12816e-03f, 3.40826e-03f, 1.68837e-03f, 1.43350e-03f,
    3.11242e-03f, 4.79133e-03f, 6.40944e-03f, 4.77052e-03f, 3.13161e-03f, 1.49269e-03f,
    2.93236e-05f, 1.62919e-03f, 3.22906e-03f, 4.82892e-03f, 6.14671e-03f, 4.58497e-03f,
    3.02322e-03f, 1.46147e-03f, 1.36017e-04f, 1.66056e-03f, 3.18509e-03f, 4.70963e-03f,
    6.04072e-03f, 4.55251e-03f, 3.06429e-03f, 1.57608e-03f, 8.78619e-05f, 9.32810e-05f,
    1.54604e-03f, 2.99880e-03f, 4.45155e-03f, 5.90431e-03f, 4.65566e-03f, 3.23752e-03f,
    1.81937e-03f, 4.01226e-04f, 1.30263e-03f, 2.68698e-03f, 4.07134e-03f, 5.45570e-03f,
    4.87832e-03f, 3.52695e-03f, 2.17558e-03f, 8.24205e-04f, 9.45950e-04f, 2.26513e-03f,
    3.58430e-03f, 4.90348e-03f, 5.20570e-03f, 3.91795e-03f, 2.63021e-03f, 1.34246e-03f,
    5.47149e-05f, 4.90379e-04f, 1.74744e-03f, 3.00451e-03f, 4.26157e-03f, 5.51864e-03f,
    4.39707e-03f, 3.16996e-03f, 1.94284e-03f, 7.15731e-04f, 1.14698e-03f, 2.34486e-03f,
    3.54273e-03f, 4.74061e-03f, 4.95198e-03f, 3.78265e-03f, 2.61331e-03f, 1.44397e-03f,
    2.74637e-04f, 4.75695e-04f, 1.61717e-03f, 2.75865e-03f, 3.90013e-03f, 5.04160e-03f,
    4.45712e-03f, 3.34284e-03f, 2.22856e-03f, 1.11428e-03f};

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

std::vector<float> mel128FilterBank(lut::Span<const float> input) {
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
  std::vector<float> melFbank = mel128FilterBank(magnitudes);

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

  int numFrames = inputs.getShape(0) / HopLength;
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
