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

#include "ly/internal/tensor_data.h"

#include <stdint.h>
#include <memory>

#include "ly/device.h"
#include "ly/dtype.h"
#include "lyutil/error.h"
#include "lyutil/platform.h"
#include "lyutil/span.h"

namespace ly {
namespace internal {

void TensorData::throwIfInvalid() {
  int16_t dtype = getDType();
  switch (dtype) {
    case DType::kUnknown:
      throw lut::AbortedError("invalid tensor (dtype=unknown).");
      break;
    case DType::kQ4:
      if (getSlot(1)->getDType() != DType::kFloat16 || getSlot(2)->getDType() != DType::kUInt8) 
        throw lut::AbortedError("invalid q4 tensor data type.");
      if (getNumEl() / getDType().getGroupSize() != getSlot(1)->getNumEl())
        throw lut::AbortedError("tensor data and scale size mismatch.");
      if ((getNumEl() / getDType().getGroupSize() + 1) / 2 != getSlot(2)->getNumEl())
        throw lut::AbortedError("tensor data and zero-point size mismatch.");
      break;
  }
}

}  // namespace internal
}  // namespace ly
