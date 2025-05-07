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

#include "lutil/internal/base_array.h"

namespace lut {

template<typename T>
class FixedArray : public internal::BaseArray<T> {
 public:
  FixedArray() noexcept
      : internal::BaseArray<T>() {
  }
  FixedArray(int64_t size) noexcept
      : internal::BaseArray<T>(size ? new T[size] : nullptr, size) {
  }
  ~FixedArray() noexcept {
    delete[] internal::BaseArray<T>::_ptr;
    internal::BaseArray<T>::_ptr = nullptr;
    internal::BaseArray<T>::_size = 0;
  }

  // copy
  FixedArray(FixedArray<T> &) = delete;
  FixedArray<T> &operator=(FixedArray<T> &) = delete;

  // move
  FixedArray(FixedArray<T> &&array) noexcept {
    internal::BaseArray<T>::_ptr = array._ptr;
    internal::BaseArray<T>::_size = array._size;
    array._ptr = nullptr;
    array._size = 0;
  }
  FixedArray<T> &operator=(FixedArray<T> &&array) noexcept {
    if (internal::BaseArray<T>::_ptr) {
      delete[] internal::BaseArray<T>::_ptr;
    }

    internal::BaseArray<T>::_ptr = array._ptr;
    internal::BaseArray<T>::_size = array._size;
    array._ptr = nullptr;
    array._size = 0;

    return *this;
  }

  FixedArray<T> copy() const {
    FixedArray<T> l(internal::BaseArray<T>::_size);
    std::copy(internal::BaseArray<T>::begin(), internal::BaseArray<T>::end(), l.begin());

    return l;
  }
};

}  // namespace lut
