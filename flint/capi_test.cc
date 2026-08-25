// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "flint/capi.h"

#include <string>
#include <vector>

#include "catch2/catch_amalgamated.hpp"

namespace {

/// Destroys a handle however the test leaves the scope, so a failed assertion does not leak.
class ScopedTensor {
 public:
  ScopedTensor() = default;
  ScopedTensor(const ScopedTensor &) = delete;
  ScopedTensor &operator=(const ScopedTensor &) = delete;

  ~ScopedTensor() {
    fl_tensor_destroy(_tensor);
  }

  fl_tensor_t *operator&() {
    return &_tensor;
  }

  operator fl_tensor_t() const {
    return _tensor;
  }

 private:
  fl_tensor_t _tensor = nullptr;
};

/// Creates a CPU float tensor holding `data`, the input every operator test starts from.
void makeFloats(const std::vector<int32_t> &shape, const std::vector<float> &data,
                fl_tensor_t *out) {
  CATCH_REQUIRE(
      fl_tensor_from_data(
          shape.data(),
          static_cast<int32_t>(shape.size()),
          FL_DTYPE_FLOAT,
          data.data(),
          static_cast<int64_t>(data.size() * sizeof(float)),
          out) == FL_OK);
}

std::vector<float> readFloats(fl_tensor_t tensor) {
  int64_t nbytes = 0;
  CATCH_REQUIRE(fl_tensor_get_nbytes(tensor, &nbytes) == FL_OK);

  std::vector<float> values(static_cast<size_t>(nbytes) / sizeof(float));
  CATCH_REQUIRE(fl_tensor_copy_to_host(tensor, values.data(), nbytes) == FL_OK);
  return values;
}

}  // namespace

CATCH_TEST_CASE("flint C API reports tensor metadata", "[core][flint][capi]") {
  fl_init();
  CATCH_REQUIRE(fl_get_last_error_code() == FL_OK);

  const std::vector<int32_t> shape{2, 3};
  ScopedTensor tensor;
  CATCH_REQUIRE(
      fl_tensor_zeros(shape.data(), 2, FL_DTYPE_FLOAT, FL_DEVICE_CPU, &tensor) == FL_OK);

  int32_t dim = 0;
  CATCH_REQUIRE(fl_tensor_get_dim(tensor, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 2);

  int32_t size = 0;
  CATCH_REQUIRE(fl_tensor_get_shape(tensor, 0, &size) == FL_OK);
  CATCH_REQUIRE(size == 2);
  CATCH_REQUIRE(fl_tensor_get_shape(tensor, -1, &size) == FL_OK);
  CATCH_REQUIRE(size == 3);

  int64_t numel = 0;
  CATCH_REQUIRE(fl_tensor_get_numel(tensor, &numel) == FL_OK);
  CATCH_REQUIRE(numel == 6);

  fl_dtype_t dtype = FL_DTYPE_UNKNOWN;
  CATCH_REQUIRE(fl_tensor_get_dtype(tensor, &dtype) == FL_OK);
  CATCH_REQUIRE(dtype == FL_DTYPE_FLOAT);

  fl_device_type_t device = FL_DEVICE_UNKNOWN;
  CATCH_REQUIRE(fl_tensor_get_device(tensor, &device) == FL_OK);
  CATCH_REQUIRE(device == FL_DEVICE_CPU);

  int32_t contiguous = 0;
  CATCH_REQUIRE(fl_tensor_is_contiguous(tensor, &contiguous) == FL_OK);
  CATCH_REQUIRE(contiguous == 1);

  CATCH_REQUIRE(readFloats(tensor) == std::vector<float>(6, 0.0f));
}

CATCH_TEST_CASE("flint C API round-trips element data", "[core][flint][capi]") {
  fl_init();

  const std::vector<int32_t> shape{2, 2};
  const std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
  ScopedTensor tensor;
  CATCH_REQUIRE(
      fl_tensor_from_data(
          shape.data(),
          2,
          FL_DTYPE_FLOAT,
          data.data(),
          static_cast<int64_t>(data.size() * sizeof(float)),
          &tensor) == FL_OK);

  CATCH_REQUIRE(readFloats(tensor) == data);
}

CATCH_TEST_CASE("flint C API packs a transposed tensor on copy", "[core][flint][capi]") {
  fl_init();

  const std::vector<int32_t> shape{2, 2};
  const std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
  ScopedTensor tensor;
  CATCH_REQUIRE(
      fl_tensor_from_data(
          shape.data(),
          2,
          FL_DTYPE_FLOAT,
          data.data(),
          static_cast<int64_t>(data.size() * sizeof(float)),
          &tensor) == FL_OK);

  ScopedTensor transposed;
  CATCH_REQUIRE(fl_tensor_transpose(tensor, 0, 1, &transposed) == FL_OK);

  int32_t contiguous = 1;
  CATCH_REQUIRE(fl_tensor_is_contiguous(transposed, &contiguous) == FL_OK);
  CATCH_REQUIRE(contiguous == 0);

  // The copy has to make it contiguous, so the caller sees the transposed order.
  CATCH_REQUIRE(readFloats(transposed) == std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f});
}

CATCH_TEST_CASE("flint C API reshapes and slices a tensor", "[core][flint][capi]") {
  fl_init();

  const std::vector<int32_t> shape{6};
  const std::vector<float> data{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  ScopedTensor tensor;
  CATCH_REQUIRE(
      fl_tensor_from_data(
          shape.data(),
          1,
          FL_DTYPE_FLOAT,
          data.data(),
          static_cast<int64_t>(data.size() * sizeof(float)),
          &tensor) == FL_OK);

  const std::vector<int32_t> viewShape{2, 3};
  ScopedTensor view;
  CATCH_REQUIRE(fl_tensor_view(tensor, viewShape.data(), 2, &view) == FL_OK);
  int32_t dim = 0;
  CATCH_REQUIRE(fl_tensor_get_dim(view, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 2);

  ScopedTensor row;
  CATCH_REQUIRE(fl_tensor_subtensor(view, 1, &row) == FL_OK);
  CATCH_REQUIRE(readFloats(row) == std::vector<float>{3.0f, 4.0f, 5.0f});

  // FL_NONE leaves that end of the range where it is.
  ScopedTensor tail;
  CATCH_REQUIRE(fl_tensor_slice(tensor, 0, 4, FL_NONE, &tail) == FL_OK);
  CATCH_REQUIRE(readFloats(tail) == std::vector<float>{4.0f, 5.0f});

  ScopedTensor unsqueezed;
  CATCH_REQUIRE(fl_tensor_unsqueeze(tensor, 0, &unsqueezed) == FL_OK);
  CATCH_REQUIRE(fl_tensor_get_dim(unsqueezed, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 2);

  ScopedTensor squeezed;
  CATCH_REQUIRE(fl_tensor_squeeze(unsqueezed, 0, &squeezed) == FL_OK);
  CATCH_REQUIRE(fl_tensor_get_dim(squeezed, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 1);
}

CATCH_TEST_CASE("flint C API shares storage between handles", "[core][flint][capi]") {
  fl_init();

  const std::vector<int32_t> shape{2};
  const std::vector<float> data{7.0f, 8.0f};
  ScopedTensor tensor;
  CATCH_REQUIRE(
      fl_tensor_from_data(
          shape.data(),
          1,
          FL_DTYPE_FLOAT,
          data.data(),
          static_cast<int64_t>(data.size() * sizeof(float)),
          &tensor) == FL_OK);

  ScopedTensor clone;
  CATCH_REQUIRE(fl_tensor_clone(tensor, &clone) == FL_OK);
  CATCH_REQUIRE(readFloats(clone) == data);

  // Destroying one handle leaves the storage alive for the other.
  fl_tensor_destroy(tensor);
  *(&tensor) = nullptr;
  CATCH_REQUIRE(readFloats(clone) == data);
}

CATCH_TEST_CASE("flint C API reports errors instead of throwing", "[core][flint][capi]") {
  fl_init();

  const std::vector<int32_t> shape{2, 2};
  const std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};

  // A size that does not match the shape is rejected before anything is allocated.
  ScopedTensor mismatched;
  CATCH_REQUIRE(
      fl_tensor_from_data(shape.data(), 2, FL_DTYPE_FLOAT, data.data(), 8, &mismatched) ==
      FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_get_last_error_code() == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(fl_get_last_error_message()).find("data_size") != std::string::npos);

  ScopedTensor invalidDtype;
  CATCH_REQUIRE(
      fl_tensor_zeros(shape.data(), 2, FL_DTYPE_UNKNOWN, FL_DEVICE_CPU, &invalidDtype) ==
      FL_ERROR_INVALID_ARG);

  int32_t dim = 0;
  CATCH_REQUIRE(fl_tensor_get_dim(nullptr, &dim) == FL_ERROR_INVALID_ARG);

  ScopedTensor tensor;
  CATCH_REQUIRE(
      fl_tensor_zeros(shape.data(), 2, FL_DTYPE_FLOAT, FL_DEVICE_CPU, &tensor) == FL_OK);
  CATCH_REQUIRE(fl_get_last_error_code() == FL_OK);

  std::vector<float> tooSmall(2);
  CATCH_REQUIRE(
      fl_tensor_copy_to_host(tensor, tooSmall.data(), 8) == FL_ERROR_INVALID_ARG);

  // Destroying a null handle is allowed, which is what lets callers clean up unconditionally.
  fl_tensor_destroy(nullptr);
}

CATCH_TEST_CASE("flint C API runs the functional operators", "[core][flint][capi]") {
  fl_init();

  ScopedTensor a;
  ScopedTensor b;
  makeFloats({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &a);
  makeFloats({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f}, &b);

  ScopedTensor sum;
  CATCH_REQUIRE(fl_add(a, b, &sum) == FL_OK);
  CATCH_REQUIRE(readFloats(sum) == std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f});

  ScopedTensor scaled;
  CATCH_REQUIRE(fl_mul_scalar(a, 2.0f, &scaled) == FL_OK);
  CATCH_REQUIRE(readFloats(scaled) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});

  ScopedTensor identity;
  makeFloats({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}, &identity);
  ScopedTensor product;
  CATCH_REQUIRE(fl_matmul(a, identity, &product) == FL_OK);
  CATCH_REQUIRE(readFloats(product) == readFloats(a));

  // The rows of a softmax sum to one, which is what fl_sum() should find.
  ScopedTensor probabilities;
  ScopedTensor rowSums;
  ScopedTensor ones;
  CATCH_REQUIRE(fl_softmax(a, &probabilities) == FL_OK);
  CATCH_REQUIRE(fl_sum(probabilities, -1, &rowSums) == FL_OK);
  makeFloats({2}, {1.0f, 1.0f}, &ones);

  int32_t close = 0;
  CATCH_REQUIRE(fl_all_close(rowSums, ones, 1e-3f, 1e-5f, &close) == FL_OK);
  CATCH_REQUIRE(close == 1);

  ScopedTensor filled;
  CATCH_REQUIRE(fl_tensor_zeros(std::vector<int32_t>{2}.data(), 1, FL_DTYPE_FLOAT, FL_DEVICE_CPU,
                                &filled) == FL_OK);
  CATCH_REQUIRE(fl_fill(filled, 3.0f) == FL_OK);
  CATCH_REQUIRE(readFloats(filled) == std::vector<float>{3.0f, 3.0f});

  // A null handle is still reported rather than dereferenced.
  ScopedTensor unused;
  CATCH_REQUIRE(fl_add(nullptr, b, &unused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_fill(nullptr, 1.0f) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_causal_mask(4, FL_DEVICE_UNKNOWN, &unused) == FL_ERROR_INVALID_ARG);
}
