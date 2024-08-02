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

#include <functional>

#include "lut/log.h"
#include "lut/noncopyable.h"

namespace lut {

// Stores the C pointer and it's destroy function
template<typename T>
class c_ptr : private NonCopyable {
 public:
  c_ptr();
  c_ptr(T *ptr, std::function<void(T *)> deleter);
  c_ptr(c_ptr<T> &&auto_cptr) noexcept;
  ~c_ptr();

  c_ptr<T> &operator=(c_ptr<T> &&auto_cptr);

  // return the pointer and release the ownership
  T *Release();

  // get the pointer
  T *get() {
    return _ptr;
  }
  const T *get() const {
    return _ptr;
  }

  // get the pointer to this pointer.
  T **get_pp() {
    CHECK(_deleter);
    return &_ptr;
  }

 private:
  T *_ptr;
  std::function<void(T *)> _deleter;
};

template<typename T>
inline c_ptr<T>::c_ptr()
    : _ptr(nullptr),
      _deleter(nullptr) {
}
template<typename T>
inline c_ptr<T>::c_ptr(T *ptr, std::function<void(T *)> deleter)
    : _ptr(ptr),
      _deleter(deleter) {
}
template<typename T>
inline c_ptr<T>::c_ptr(c_ptr<T> &&auto_cptr) noexcept
    : _ptr(auto_cptr._ptr),
      _deleter(auto_cptr._deleter) {
  auto_cptr._ptr = nullptr;
}
template<typename T>
inline c_ptr<T>::~c_ptr() {
  if (_ptr) {
    _deleter(_ptr);
    _ptr = nullptr;
  }
}
template<typename T>
c_ptr<T> &c_ptr<T>::operator=(c_ptr<T> &&auto_cptr) {
  if (_ptr) {
    _deleter(_ptr);
    _ptr = nullptr;
  }

  _ptr = auto_cptr._ptr;
  _deleter = auto_cptr._deleter;

  auto_cptr._ptr = nullptr;
  return *this;
}
template<typename T>
inline T *c_ptr<T>::Release() {
  T *ptr = _ptr;
  _ptr = nullptr;

  return ptr;
}

}  // namespace lut
