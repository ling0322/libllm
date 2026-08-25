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

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toCuda(const Tensor &a) {
  return F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

}  // namespace

CATCH_TEST_CASE("test CUDA lookup", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor embd = F::rand({10, 32}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor xr = F::lookup(embd, ids);

  Tensor x = F::lookup(toCuda(embd), F::to(Device::getCuda(), ids));

  CATCH_REQUIRE(F::allClose(toCpu(x), xr));

  // packed indices are 1D and give one embedding row per index.
  Tensor packedIds = Tensor::create<LongType>({3}, {1, 2, 3});
  Tensor packedRef = F::lookup(embd, packedIds);
  Tensor packed = F::lookup(toCuda(embd), F::to(Device::getCuda(), packedIds));

  CATCH_REQUIRE(packed.getShape() == std::vector<int>{3, 32});
  CATCH_REQUIRE(F::allClose(toCpu(packed), packedRef));
}

CATCH_TEST_CASE("test CUDA lookup (embedding widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A row is copied by a grid that is 256 threads wide, so widths on both sides of the block
  // boundary decide whether the tail of a row is reached at all.
  for (int width : {1, 255, 256, 257, 1000}) {
    Tensor embd = F::rand({6, width}, DType::kFloat);
    Tensor ids = Tensor::create<LongType>({2, 3}, {0, 1, 2, 3, 4, 5});

    Tensor x = F::lookup(toCuda(embd), F::to(Device::getCuda(), ids));
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{2, 3, width});
    CATCH_REQUIRE(F::allClose(toCpu(x), F::lookup(embd, ids), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA lookup (index edges)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The first and last rows of the table are where an off-by-one in the row offset shows up, and
  // a repeated index must return the same row every time rather than advancing.
  Tensor embd = F::rand({5, 8}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 3}, {0, 4, 0, 4, 2, 2});

  Tensor x = F::lookup(toCuda(embd), F::to(Device::getCuda(), ids));
  CATCH_REQUIRE(F::allClose(toCpu(x), F::lookup(embd, ids), 5e-3));

  // a table with a single row, so every index has to resolve to it.
  Tensor single = F::rand({1, 8}, DType::kFloat);
  Tensor zeros = Tensor::create<LongType>({3}, {0, 0, 0});
  Tensor y = F::lookup(toCuda(single), F::to(Device::getCuda(), zeros));
  CATCH_REQUIRE(y.getShape() == std::vector<int>{3, 8});
  CATCH_REQUIRE(F::allClose(toCpu(y), F::lookup(single, zeros), 5e-3));
}

CATCH_TEST_CASE("test CUDA lookup (float table)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The operator has a separate instantiation for a float table; moving the table across
  // without casting is what selects it.
  Tensor embd = F::rand({6, 16}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 2}, {0, 5, 3, 1});

  Tensor table = F::to(Device::getCuda(), embd);
  CATCH_REQUIRE(table.getDType() == DType::kFloat);

  Tensor x = F::lookup(table, F::to(Device::getCuda(), ids));
  CATCH_REQUIRE(x.getDType() == DType::kFloat);
  CATCH_REQUIRE(F::allClose(F::to(Device::getCpu(), x), F::lookup(embd, ids)));

  // and the packed 1D form of the same table.
  Tensor packedIds = Tensor::create<LongType>({2}, {4, 2});
  Tensor packed = F::lookup(table, F::to(Device::getCuda(), packedIds));
  CATCH_REQUIRE(packed.getShape() == std::vector<int>{2, 16});
  CATCH_REQUIRE(F::allClose(F::to(Device::getCpu(), packed), F::lookup(embd, packedIds)));
}

}  // namespace fl
