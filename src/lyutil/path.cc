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

#include "lyutil/path.h"

#include "lyutil/log.h"
#include "lyutil/platform.h"
#include "lyutil/strings.h"

namespace ly {

Path::Path(const std::string &path) : _path(normPath(path)) {}
Path::Path(std::string &&path): _path(normPath(path)) {}

Path Path::dirname() const {
  const char *delim = getPathDelim();
  size_t lastDelimIdx = _path.find_last_of(delim);
  if (lastDelimIdx == std::string::npos) {
    return Path();
  }

  std::string name = std::string(_path.begin(), _path.begin() + lastDelimIdx);
  name = trimRight(name, delim);
  return name;
}

Path Path::basename() const {
  const char *delim = getPathDelim();
  size_t lastDelimIdx = _path.find_last_of(delim);
  if (lastDelimIdx == std::string::npos) {
    return _path;
  }

  return std::string(_path.begin() + lastDelimIdx + 1, _path.end());
}

bool Path::operator==(const Path &r) const {
  return _path == r._path;
}

bool Path::operator==(const std::string &r) const {
  return _path == r;
}

Path Path::operator/(const Path &path) const {
  const char *delim = getPathDelim();
  std::string left = _path;
  if (left.empty()) {
    return path;
  }

  if ((!left.empty()) && left.back() == delim[0]) {
    left = trimRight(left, delim);
  }

  std::string right = path._path;
  if ((!right.empty()) && right.front() == delim[0]) {
    right = trimLeft(right, delim);
  }

  return left + delim + right;
}

Path Path::operator/(const std::string &path) const {
  return *this / Path(path);
}

std::string Path::string() const {
  return _path;
}

std::wstring Path::wstring() const {
  return toWide(_path);
}

} // namespace ly
