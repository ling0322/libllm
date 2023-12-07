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

#include "ly/device.h"

#include "lyutil/log.h"
#include "ly/operators/cuda/cuda_operators.h"

namespace ly {

Device::Device() : _type(Type::kUnknown) {}
Device::Device(Type type) : _type(type) {}

Device Device::getCpu() {
  return Device(Type::kCpu);
}

Device Device::getCuda() {
  return Device(Type::kCuda);
}

bool Device::isCudaAvailable() {
#ifdef LLYN_CUDA_ENABLED
  return op::cuda::CudaOperators::isAvailable();
#else
  return false;
#endif
}

std::string Device::getName() const {
  switch (_type) {
    case kCpu:
      return "cpu";
    case kCuda:
      return "cuda";
    default:
      NOT_IMPL();
  }
}

}  // namespace ly
