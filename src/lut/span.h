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

#include <array>

#include "lut/attributes.h"
#include "lut/fixed_array.h"
#include "lut/internal/base_array.h"
#include "lut/log.h"

namespace lut {

template<typename T>
class Span : public internal::BaseArray<T> {
 public:
  constexpr Span() noexcept
      : internal::BaseArray<T>() {
  }
  constexpr Span(T *ptr, typename internal::BaseArray<T>::size_type size)
      : internal::BaseArray<T>(ptr, size) {
  }

  // automatic convert initializer_list to Span<const T>.
  // NOTE: initializer_list should outlives span when using this constructor.
  // Examples:
  //   Span<const int> v = {1, 2, 3};  // WRONG: lifetime of initializer_list is shorter than v;
  template<typename U = T, typename = typename std::enable_if<std::is_const<T>::value, U>::type>
  constexpr Span(std::initializer_list<typename internal::BaseArray<T>::value_type> v
                     LUT_LIFETIME_BOUND) noexcept
      : Span(v.begin(), v.size()) {
  }

  // automatic convert std::vector<T> to Span<const T>.
  // NOTE: initializer_list should outlives span when using this constructor.
  // Examples:
  //   Span<const int> v = {1, 2, 3};  // WRONG: lifetime of initializer_list is shorter than v;
  template<typename U = T, typename = typename std::enable_if<std::is_const<T>::value, U>::type>
  constexpr Span(
      const std::vector<typename internal::BaseArray<T>::value_type> &v LUT_LIFETIME_BOUND) noexcept
      : Span(v.data(), v.size()) {
  }

  // automatic convert std::array<T> to Span<const T>.
  // NOTE: initializer_list should outlives span when using this constructor.
  // Examples:
  //   Span<const int> v = {1, 2, 3};  // WRONG: lifetime of initializer_list is shorter than v;
  template<
      typename U = T,
      std::size_t N,
      typename = typename std::enable_if<std::is_const<T>::value, U>::type>
  constexpr Span(const std::array<typename internal::BaseArray<T>::value_type, N> &v
                     LUT_LIFETIME_BOUND) noexcept
      : Span(v.data(), v.size()) {
  }

  constexpr Span<T> subspan(
      typename internal::BaseArray<T>::size_type pos = 0,
      typename internal::BaseArray<T>::size_type len = internal::BaseArray<T>::npos) const {
    CHECK(pos <= internal::BaseArray<T>::size());
    len = std::min(internal::BaseArray<T>::size() - pos, len);
    return Span<T>(internal::BaseArray<T>::data() + pos, len);
  }
};

template<typename T>
constexpr Span<T> makeSpan(T *ptr, typename Span<T>::size_type size) {
  return Span<T>(ptr, size);
}
template<typename T>
constexpr Span<const T> makeConstSpan(const T *ptr, typename Span<T>::size_type size) {
  return Span<const T>(ptr, size);
}
template<typename T>
constexpr Span<T> makeSpan(std::vector<T> &v) {
  return Span<T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(const std::vector<T> &v) {
  return Span<const T>(v.data(), v.size());
}

template<typename T>
constexpr Span<T> makeSpan(const FixedArray<T> &v) {
  return Span<T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(const FixedArray<T> &v) {
  return Span<const T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(std::initializer_list<T> v) {
  return Span<const T>(v.begin(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(Span<T> v) {
  return Span<const T>(v.data(), v.size());
}

}  // namespace lut
