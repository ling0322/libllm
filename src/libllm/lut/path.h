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

#include <string>

namespace lut {

// provide base functions for path. For example, path::Join, dirname, basename
// and convert to wstring, ...
class Path {
 public:
  static Path currentModulePath();
  static Path currentExecutablePath();

  Path() = default;
  Path(const std::string &path);
  Path(std::string &&path);

  bool operator==(const Path &r) const;
  bool operator==(const std::string &r) const;

  Path dirname() const;
  Path basename() const;
  std::string extension() const;
  bool isabs() const;

  Path operator/(const Path &path) const;
  Path operator/(const std::string &path) const;

  std::string string() const;
  std::wstring wstring() const;

 private:
  std::string _path;

  // normalize path string.
  static std::string normPath(const std::string &path);
};

}  // namespace lut
