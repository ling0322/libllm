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

#include <memory>
#include <string>

#include "lutil/noncopyable.h"

namespace lut {

// Dynamic library loader. It stores the instance of dynamic library and
// supports get function from it.
//
// Example:
//
// auto library = SharedLibrary::open("foo");  // foo.dll for Windows
// std::function<int(float)> func = library.GetFunc<int(float)>("bar");
class SharedLibrary : private NonCopyable {
 public:
  class Impl;

  ~SharedLibrary();

  // load a library by name from OS. Firstly, it will search the same directory
  // as caller module. Then, fallback to system search. In windows, the actual
  // library name would be `name`.dll. In Linux, it would be lib`name`.so
  static std::unique_ptr<SharedLibrary> open(const std::string &name);

  // get function by name. return nullptr if function not found
  template<typename T>
  T getFunc(const std::string &name) {
    return reinterpret_cast<T>(getFuncPtr(name));
  }

 private:
  std::unique_ptr<Impl> _impl;

  SharedLibrary();

  // get raw function pointer by name. throw error if function not exist or
  // other errors occured
  void *getFuncPtr(const std::string &name);
};

}  // namespace lut
