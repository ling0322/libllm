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

#include "catch2/catch_amalgamated.hpp"
#include "libllm/lut/path.h"
#include "libllm/lut/platform.h"

namespace lut {

Path toPath(const char *pcs) {
  std::string s = pcs;
#ifdef LUT_PLATFORM_WINDOWS
  for (char &ch : s) {
    if (ch == '/') ch = '\\';
  }
#endif

  return s;
}

CATCH_TEST_CASE("get current module path success", "[core][util]") {
  Path current_module_path = Path::currentModulePath();
  CATCH_REQUIRE(!current_module_path.string().empty());
}

CATCH_TEST_CASE("get current executable path success", "[core][util]") {
  Path current_module_path = Path::currentExecutablePath();
  CATCH_REQUIRE(!current_module_path.string().empty());
}

CATCH_TEST_CASE("path operations works", "[core][util]") {
  CATCH_REQUIRE(toPath("foo") / toPath("bar.txt") == toPath("foo/bar.txt"));
  CATCH_REQUIRE(toPath("foo/") / toPath("bar.txt") == toPath("foo/bar.txt"));
  CATCH_REQUIRE(toPath("foo//") / toPath("bar.txt") == toPath("foo/bar.txt"));
  CATCH_REQUIRE(toPath("foo") / toPath("/bar.txt") == toPath("foo/bar.txt"));
  CATCH_REQUIRE(toPath("foo") / toPath("//bar.txt") == toPath("foo/bar.txt"));
  CATCH_REQUIRE(toPath("foo//") / toPath("//bar.txt") == toPath("foo/bar.txt"));
  CATCH_REQUIRE(toPath("foo//") / toPath("") == toPath("foo/"));
  CATCH_REQUIRE(toPath("") / toPath("bar.txt") == toPath("bar.txt"));
  CATCH_REQUIRE(toPath("") / toPath("/bar.txt") == toPath("/bar.txt"));

  CATCH_REQUIRE(toPath("foo/bar.txt").basename() == toPath("bar.txt"));
  CATCH_REQUIRE(toPath("foo/bar.txt").dirname() == toPath("foo"));
  CATCH_REQUIRE(toPath("baz/foo/bar.txt").dirname() == toPath("baz/foo"));
  CATCH_REQUIRE(toPath("bar.txt").dirname() == toPath(""));
  CATCH_REQUIRE(toPath("foo/").basename() == toPath(""));
  CATCH_REQUIRE(toPath("foo//").basename() == toPath(""));

  CATCH_REQUIRE(toPath("").basename() == toPath(""));
  CATCH_REQUIRE(toPath("").dirname() == toPath(""));
}

}  // namespace lut
