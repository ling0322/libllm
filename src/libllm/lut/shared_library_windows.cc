// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include "libllm/lut/shared_library.h"

#include <functional>
#include <memory>
#include <windows.h>

#include "libllm/lut/error.h"
#include "libllm/lut/log.h"
#include "libllm/lut/path.h"
#include "libllm/lut/strings.h"

namespace lut {

class SharedLibrary::Impl {
 public:
  ~Impl();

  static std::unique_ptr<Impl> open(const std::string &name);
  void *getFuncPtr(const std::string &filename);

 private:
  HMODULE _module;

  Impl();
};

SharedLibrary::Impl::Impl() : _module(nullptr) {}
SharedLibrary::Impl::~Impl() {
  if (_module) {
    FreeLibrary(_module);
    _module = nullptr;
  }
}

std::unique_ptr<SharedLibrary::Impl> SharedLibrary::Impl::open(const std::string &name) {
  std::unique_ptr<Impl> impl{new Impl()};
  Path filename = std::string(name) + ".dll";

  // first try to load the dll from same folder as current module
  Path modulePath = Path::currentModulePath();
  modulePath = modulePath.dirname();
  Path absFilename = modulePath / filename;

  DWORD code =  S_OK;
  impl->_module = LoadLibraryW(absFilename.wstring().c_str());
  if (!impl->_module) {
    code = GetLastError();
    LOG(DEBUG) << "Load library " << absFilename.string() << " failed with code " << code
               << " fall back to system search.";

    // fallback to system search
    impl->_module = LoadLibraryW(filename.wstring().c_str());
    if (!impl->_module) {
      code = GetLastError();
      throw AbortedError(lut::sprintf(
          "Load library %s failed with code 0x%x.", absFilename.string(), code));
    }
  }

  return impl;
}

void *SharedLibrary::Impl::getFuncPtr(const std::string &func_name) {
  CHECK(_module) << "call GetRawFuncPtr() on empty SharedLibrary";
  FARPROC func = GetProcAddress(_module, std::string(func_name).c_str());
  return reinterpret_cast<void *>(func);
}

// -- class SharedLibrary ----------

SharedLibrary::SharedLibrary() {}
SharedLibrary::~SharedLibrary() {}

std::unique_ptr<SharedLibrary> SharedLibrary::open(const std::string &name) {
  std::unique_ptr<SharedLibrary> library{new SharedLibrary()};
  library->_impl = Impl::open(name);
  return library;
}
void *SharedLibrary::getFuncPtr(const std::string &name) {
  return _impl->getFuncPtr(name);
}

} // namespace lut
