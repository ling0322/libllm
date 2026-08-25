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
#include <cstdint>

#include "catch2/catch_amalgamated.hpp"
#include "flint/cuda/common.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/memory.h"

namespace fl {

CATCH_TEST_CASE("test CUDA memory snapshot", "[fl][cuda][memory]") {
  MemorySnapshot::resetPeakStats(Device::getCuda());

  MemorySnapshot before = MemorySnapshot::capture(Device::getCuda());
  CATCH_REQUIRE(before.getTotalMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() <= before.getTotalMemory());

  int64_t bytes = 0;
  {
    Tensor x = F::tensor({1024, 1024}, DType::kFloat16, Device::getCuda());
    bytes = x.getNumEl() * 2;

    MemorySnapshot allocated = MemorySnapshot::capture(Device::getCuda());
    CATCH_REQUIRE(allocated.getAllocatedMemory() >= before.getAllocatedMemory() + bytes);
  }

  // the tensor is gone, but its bytes stay in the pool and remain in the peak.
  MemorySnapshot after = MemorySnapshot::capture(Device::getCuda());
  CATCH_REQUIRE(after.getAllocatedMemory() <= before.getAllocatedMemory());
  CATCH_REQUIRE(after.getPeakAllocatedMemory() >= bytes);
}

CATCH_TEST_CASE("test CUDA FastDivmod", "[fl][cuda]") {
  constexpr uint32_t divisors[] = {1, 2, 3, 7, 16, 255, 65535, INT32_MAX};

  for (uint32_t divisor : divisors) {
    op::cuda::FastDivmod divider(divisor);
    uint32_t dividends[] = {
        0, 1, divisor - 1, divisor, std::min(divisor + 1, uint32_t{INT32_MAX}), INT32_MAX};

    for (uint32_t dividend : dividends) {
      uint32_t quotient;
      uint32_t remainder;
      divider.divmod(dividend, quotient, remainder);
      CATCH_REQUIRE(quotient == dividend / divisor);
      CATCH_REQUIRE(remainder == dividend % divisor);
    }
  }
}

CATCH_TEST_CASE("test CUDA FastDivmod (powers of two)", "[fl][cuda]") {
  // The magic-number derivation shifts by ceil(log2(divisor)); an exact power of two is where
  // that shift lands on the boundary and the multiplier is at its smallest.
  for (int shift = 0; shift < 31; ++shift) {
    uint32_t divisor = uint32_t{1} << shift;
    op::cuda::FastDivmod divider(divisor);

    uint32_t dividends[] = {
        0,
        1,
        divisor - 1,
        divisor,
        divisor + 1,
        divisor * 2 - 1,
        INT32_MAX - 1,
        INT32_MAX};
    for (uint32_t dividend : dividends) {
      uint32_t quotient;
      uint32_t remainder;
      divider.divmod(dividend, quotient, remainder);
      CATCH_INFO("divisor = " << divisor << ", dividend = " << dividend);
      CATCH_REQUIRE(quotient == dividend / divisor);
      CATCH_REQUIRE(remainder == dividend % divisor);
    }
  }
}

CATCH_TEST_CASE("test CUDA FastDivmod (exhaustive small)", "[fl][cuda]") {
  // Small divisors are what the tensor accessors actually use (one per axis), so walk every
  // dividend/divisor pair in that range rather than sampling it.
  for (uint32_t divisor = 1; divisor <= 64; ++divisor) {
    op::cuda::FastDivmod divider(divisor);
    for (uint32_t dividend = 0; dividend < 512; ++dividend) {
      uint32_t quotient;
      uint32_t remainder;
      divider.divmod(dividend, quotient, remainder);
      CATCH_INFO("divisor = " << divisor << ", dividend = " << dividend);
      CATCH_REQUIRE(quotient == dividend / divisor);
      CATCH_REQUIRE(remainder == dividend % divisor);
    }
  }
}

}  // namespace fl
