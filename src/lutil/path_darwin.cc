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

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <sys/syslimits.h>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/path.h"
#include "lutil/strings.h"

namespace lut {

Path Path::currentExecutablePath() {
  char path[PATH_MAX + 1];
  uint32_t size = sizeof(path);
  int ret = _NSGetExecutablePath(path, &size);
  if (ret) {
    throw lut::AbortedError("failed to call _NSGetExecutablePath()");
  }

  return Path(path);
}

Path Path::currentModulePath() {
  Dl_info info;
  int success = dladdr(reinterpret_cast<const void *>(&currentModulePath), &info);
  CHECK(success);

  return Path(info.dli_fname);
}

bool Path::isabs() const {
  if (_path.size() == 0) return false;
  if (_path[0] == '/') return true;

  return false;
}

std::string Path::normPath(const std::string &path) {
  return path;
}

}  // namespace lut
