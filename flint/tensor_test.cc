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

#include "flint/tensor.h"

#include "../third_party/catch2/catch_amalgamated.hpp"
#include "flint/functional.h"

namespace fl {

CATCH_TEST_CASE("test subtensor and slice", "[core][nn][tensor]") {
  Tensor tensor = Tensor::create<float>(
      {4, 4},
      {0.0f,
       0.1f,
       0.2f,
       0.3f,
       0.4f,
       0.5f,
       0.6f,
       0.7f,
       0.8f,
       0.9f,
       1.0f,
       1.1f,
       1.2f,
       1.3f,
       1.4f,
       1.5f});

  // slice (dim 0)
  Tensor subtensor = Tensor::create<float>(
      {2, 4},
      {
          0.4f,
          0.5f,
          0.6f,
          0.7f,
          0.8f,
          0.9f,
          1.0f,
          1.1f,
      });
  CATCH_REQUIRE(F::allClose(tensor.slice({1, 3}), subtensor));

  // subtensor
  subtensor = Tensor::create<float>(
      {4},
      {
          0.4f,
          0.5f,
          0.6f,
          0.7f,
      });
  CATCH_REQUIRE(F::allClose(tensor.subtensor(1), subtensor));

  // slice (any dim)
  subtensor = Tensor::create<float>(
      {2, 2},
      {
          0.5f,
          0.6f,
          0.9f,
          1.0f,
      });
  CATCH_REQUIRE(F::allClose(tensor.slice(0, {1, 3}).slice(1, {1, 3}), subtensor));
}

CATCH_TEST_CASE("test view infers a dimension", "[core][nn][tensor]") {
  Tensor tensor = F::rand({2, 3, 4, 5}, DType::kFloat);

  // A contiguous tensor and a strided one take different code paths, and only the contiguous one
  // resolves the inferred -1 before walking the strides.
  Tensor contiguousView = tensor.view({-1, 4, 5});
  CATCH_REQUIRE(contiguousView.getShape() == std::vector<int>{6, 4, 5});
  CATCH_REQUIRE(F::allClose(contiguousView, tensor.view({6, 4, 5})));

  Tensor strided = tensor.transpose(2, 3);
  CATCH_REQUIRE(!strided.isContiguous());

  Tensor stridedView = strided.view({-1, 5, 4});
  CATCH_REQUIRE(stridedView.getShape() == std::vector<int>{6, 5, 4});
  CATCH_REQUIRE(F::allClose(stridedView, strided.view({6, 5, 4})));

  // the inferred dimension can sit anywhere in the request, not only first.
  CATCH_REQUIRE(strided.view({6, 5, -1}).getShape() == std::vector<int>{6, 5, 4});
  CATCH_REQUIRE(strided.view({6, -1, 4}).getShape() == std::vector<int>{6, 5, 4});
}

CATCH_TEST_CASE("test viewAs reads the same bytes as another type", "[core][nn][tensor]") {
  // Two floats and four halves cover the same eight bytes, so a row of one is a row of the other.
  Tensor floats = Tensor::create<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor halves = floats.viewAs(DType::kFloat16);
  CATCH_REQUIRE(halves.getDType() == DType::kFloat16);
  CATCH_REQUIRE(halves.getShape() == std::vector<int>{2, 4});
  CATCH_REQUIRE(halves.getInternalData()->getRawData() == floats.getInternalData()->getRawData());

  // A pool of bytes is what this is for: write through the view, read through the storage.
  Tensor bytes = floats.viewAs(DType::kUInt8);
  CATCH_REQUIRE(bytes.getShape() == std::vector<int>{2, 8});
  F::fill(bytes.viewAs(DType::kFloat), 0.5f);
  CATCH_REQUIRE(F::allClose(floats, Tensor::create<float>({2, 2}, {0.5f, 0.5f, 0.5f, 0.5f})));

  // A row of a larger tensor keeps its stride, in the elements of the type it is now read as, and
  // the offset it starts at moves the same way. This is the shape a pool of blocks has: one row
  // per block, part of it read as one type and part of it as another.
  Tensor pool = F::rand({4, 8}, DType::kFloat);
  Tensor tail = pool.slice(1, {4, 8}).viewAs(DType::kFloat16);
  CATCH_REQUIRE(tail.getShape() == std::vector<int>{4, 8});
  CATCH_REQUIRE(tail.getStride(0) == 16);
  CATCH_REQUIRE(!tail.isContiguous());

  // The bytes it names are the tail of each row and nothing else.
  Tensor writtenBack = tail.viewAs(DType::kFloat);
  CATCH_REQUIRE(writtenBack.getShape() == std::vector<int>{4, 4});
  CATCH_REQUIRE(writtenBack.getStride(0) == 8);
  CATCH_REQUIRE(F::allClose(writtenBack, pool.slice(1, {4, 8})));
}

}  // namespace fl
