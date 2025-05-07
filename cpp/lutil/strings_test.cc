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

#include "lutil/strings.h"

#include "../../third_party/catch2/catch_amalgamated.hpp"

namespace lut {

std::vector<std::string> V(std::initializer_list<std::string> il) {
  return il;
}

CATCH_TEST_CASE("string functions works", "[core][util]") {
  CATCH_REQUIRE(trim("  ") == "");
  CATCH_REQUIRE(trim(" \ta ") == "a");
  CATCH_REQUIRE(trim("a ") == "a");
  CATCH_REQUIRE(trim("a\t") == "a");
  CATCH_REQUIRE(trim("a") == "a");

  CATCH_REQUIRE(trimLeft(" \t") == "");
  CATCH_REQUIRE(trimLeft(" \ta") == "a");
  CATCH_REQUIRE(trimLeft(" \ta ") == "a ");
  CATCH_REQUIRE(trimLeft("a ") == "a ");

  CATCH_REQUIRE(trimRight(" \t") == "");
  CATCH_REQUIRE(trimRight("a\t ") == "a");
  CATCH_REQUIRE(trimRight(" \ta\t ") == " \ta");
  CATCH_REQUIRE(trimRight(" a") == " a");
  CATCH_REQUIRE(trimRight("a") == "a");

  CATCH_REQUIRE(split("A\tB\tC", "\t") == V({"A", "B", "C"}));
  CATCH_REQUIRE(split("A.-B.-C", ".-") == V({"A", "B", "C"}));
  CATCH_REQUIRE(split("A.B.C.", ".") == V({"A", "B", "C", ""}));
  CATCH_REQUIRE(split("..A.B", ".") == V({"", "", "A", "B"}));

  std::string s, s_ref = "vanilla\xe5\x87\xaa\xc3\xa2\xf0\x9f\x8d\xad";
  std::wstring ws, ws_ref = L"vanilla\u51ea\u00e2\U0001f36d";
  CATCH_REQUIRE(toWide(s_ref) == ws_ref);
  CATCH_REQUIRE(toUtf8(ws_ref) == s_ref);
}

CATCH_TEST_CASE("sprintf works", "[ly][strings]") {
  // BVT
  CATCH_REQUIRE(lut::sprintf("%d", 22) == "22");
  CATCH_REQUIRE(lut::sprintf("foo_%d", 22) == "foo_22");
  CATCH_REQUIRE(lut::sprintf("foo%d %s", 22, "33") == "foo22 33");

  // integer
  int i = 1234567;
  CATCH_REQUIRE(lut::sprintf("%010d", i) == "0001234567");
  CATCH_REQUIRE(lut::sprintf("%10d", i) == "   1234567");
  CATCH_REQUIRE(lut::sprintf("%x", i) == "12d687");
  CATCH_REQUIRE(lut::sprintf("%10x", i) == "    12d687");
  CATCH_REQUIRE(lut::sprintf("%X", i) == "12D687");

  // float
  double f = 123.4567;
  double g = 1.234567e8;
  CATCH_REQUIRE(lut::sprintf("%.6f", f) == "123.456700");
  CATCH_REQUIRE(lut::sprintf("%.3f", f) == "123.457");
  CATCH_REQUIRE(lut::sprintf("%9.2f", f) == "   123.46");
  CATCH_REQUIRE(lut::sprintf("%09.2f", f) == "000123.46");
  CATCH_REQUIRE(lut::sprintf("%.3e", f) == "1.235e+02");
  CATCH_REQUIRE(lut::sprintf("%.3E", f) == "1.235E+02");
  CATCH_REQUIRE(lut::sprintf("%.5g", f) == "123.46");
  CATCH_REQUIRE(lut::sprintf("%.5g", g) == "1.2346e+08");
  CATCH_REQUIRE(lut::sprintf("%.5G", g) == "1.2346E+08");

  // string
  std::string foo = "foo";
  const char* bar = "bar";
  CATCH_REQUIRE(lut::sprintf("%s", foo) == "foo");
  CATCH_REQUIRE(lut::sprintf("%s", bar) == "bar");
  CATCH_REQUIRE(lut::sprintf("%s", std::string()) == "");
  CATCH_REQUIRE(lut::sprintf("%s %s", foo, bar) == "foo bar");
  CATCH_REQUIRE(lut::sprintf("%10s", foo) == "       foo");
  CATCH_REQUIRE(lut::sprintf("%-10s", foo) == "foo       ");

  // char
  CATCH_REQUIRE(lut::sprintf("%c", 'c') == "c");

  // edge cases
  CATCH_REQUIRE(lut::sprintf("%10.2f %.3e", f, f) == "    123.46 1.235e+02");
  CATCH_REQUIRE(lut::sprintf("%%%.5e", f) == "%1.23457e+02");
  CATCH_REQUIRE(lut::sprintf("%%%d%d%d%%", 1, 2, 3) == "%123%");
  CATCH_REQUIRE(lut::sprintf("%10000000d", 22) == lut::sprintf("%200d", 22));
  CATCH_REQUIRE(lut::sprintf("%1000000000000d", 22) == lut::sprintf("%200d", 22));
  CATCH_REQUIRE(lut::sprintf("foo") == "foo");
  CATCH_REQUIRE(lut::sprintf("%%") == "%");
  CATCH_REQUIRE(lut::sprintf("") == "");

  // invalid format string
  CATCH_REQUIRE(lut::sprintf("%s_%d", "foo") == "foo_%!d(<null>)");
  CATCH_REQUIRE(lut::sprintf("%s", "foo", 22) == "foo%!_(22)");
  CATCH_REQUIRE(lut::sprintf("%d", "foo") == "%!d(foo)");
  CATCH_REQUIRE(lut::sprintf("%d_foo_%d_0", 22) == "22_foo_%!d(<null>)_0");
  CATCH_REQUIRE(lut::sprintf("%o", 22) == "%!o(22)");
  CATCH_REQUIRE(lut::sprintf("%8.3o", 22) == "%!8.3o(22)");
  CATCH_REQUIRE(lut::sprintf("%8", 22) == "%!8(22)");
  CATCH_REQUIRE(lut::sprintf("%8%", 22) == "%!8%(22)");
  CATCH_REQUIRE(lut::sprintf("%") == "%!#(<null>)");
  CATCH_REQUIRE(lut::sprintf("%", 22) == "%!(22)");
  CATCH_REQUIRE(lut::sprintf("%", 22, "foo") == "%!(22)%!_(foo)");
  CATCH_REQUIRE(lut::sprintf("%8.ad", 22) == "%!8.a(22)d");
}

}  // namespace lut
