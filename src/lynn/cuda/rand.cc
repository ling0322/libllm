// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include "lynn/cuda/rand.h"

#include <curand.h>

#include <memory>

#include "lynn/cuda/cast.h"
#include "lynn/cuda/common.h"

namespace ly {
namespace op {
namespace cuda {

class Rand::Impl {
 public:
  ~Impl();
  static std::unique_ptr<Impl> newImpl();

  Tensor randNormal(lut::Span<const int> shape);
  void setSeed(uint64_t seed);

 private:
  Impl() = default;

  curandGenerator_t gen;
};

Tensor Rand::Impl::randNormal(lut::Span<const int> shape) {
  Tensor result = createCudaTensorFloat(shape);
  curandGenerateNormal(
      gen,
      result.getInternalData()->getData<float>(),
      result.getNumEl(),
      0.0,
      1.0);
  return castFloatToHalf(result);
}

void Rand::Impl::setSeed(uint64_t seed) {
  curandSetPseudoRandomGeneratorSeed(gen, seed);
}

Rand::Impl::~Impl() {
  curandDestroyGenerator(gen);
}

std::unique_ptr<Rand::Impl> Rand::Impl::newImpl() {
  std::unique_ptr<Impl> impl{new Impl()};
  curandCreateGenerator(&impl->gen, CURAND_RNG_PSEUDO_DEFAULT);
  return impl;
}

std::shared_ptr<Rand> Rand::newRand() {
  std::shared_ptr<Rand> rand{new Rand()};
  rand->_impl = Rand::Impl::newImpl();

  return rand;
}

Tensor Rand::randNormal(lut::Span<const int> shape) {
  return _impl->randNormal(shape);
}

void Rand::setSeed(uint64_t seed) {
  _impl->setSeed(seed);
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
