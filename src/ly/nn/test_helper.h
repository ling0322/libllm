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

#pragma once

#include "ly/ly.h"
#include "ly/nn/test_helper.h"
#include "ly/internal/common.h"
#include "ly/operators/cpu/fingerprint.h"
#include "lyutil/random.h"
#include "lyutil/span.h"

namespace F = ly::functional;

namespace ly {
namespace nn {

/// @brief Helper class for Module testing.
class ModuleTester {
 public:
  static constexpr int64_t RandomSeed = 106033;

  ModuleTester(Device device, DType weightType);
  virtual ~ModuleTester() = default;

  /// @brief Generate a random float tensor with given shape and the internal random number
  /// generator. Then move it to target device and cast to default float type of that device.
  /// @param shape shape of the tensor to generate.
  /// @return generated random tensor.
  Tensor generateTensor(lut::Span<const Tensor::ShapeType> shape, float scale = 1.0);

  /// @brief Initialize a module with the internal rendom number generator. The weight in module
  /// will be initialized as type `_weightType` in device `_device`.
  /// @param module the module to initialize.
  /// @param generator the random number generator.
  void randomInit(std::shared_ptr<Module> module);

  /// @brief Get the context to create a Module.
  Context getCtx() const;

  /// @brief Copy a tensor to target device specified by `_device`, If x is a float type, it will
  /// also cast it into the default float type of the device.
  /// @param x tensor to copy.
  /// @return tensor in target device.
  Tensor toTargetDevice(Tensor x) const;

  /// @brief Copy a tensor to CPU, If x is a float type, it will also cast it into the default
  /// float type of CPU.
  /// @param x tensor to copy.
  /// @return tensor in CPU.
  Tensor toCpu(Tensor x) const;

  /// @brief Check if the data in tensor `a` is equal (with tolerance) to the elements in ref.
  /// @param a the tensor to check.
  /// @param ref the reference data.
  bool allClose(Tensor a, lut::Span<const float> ref, float atol = 1e-3, float rtol = 5e-3) const;

  
  DType getWeightType() const { return _weightType; }

 protected:
  Device _device;
  DType _weightType;
  lut::Random _random;
};

}  // namespace nn
}  // namespace ly
