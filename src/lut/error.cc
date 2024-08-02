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

#include "lut/error.h"

namespace lut {

std::string getErrorCodeName(ErrorCode code) {
  switch (code) {
    case ErrorCode::Aborted:
      return "Aborted";
    case ErrorCode::OutOfRange:
      return "OutOfRange";
    case ErrorCode::InvalidArg:
      return "InvalidArg";
    case ErrorCode::NotImplemented:
      return "NotImplemented";
    default:
      return "Unknown";
  }
}

std::string buildErrorMsg(ErrorCode code, const std::string &what) {
  return getErrorCodeName(code) + ": " + what;
}
Error::Error()
    : _code(ErrorCode::OK) {
}
Error::Error(ErrorCode code, const std::string &what)
    : _code(code),
      _what(what) {
}
Error::~Error() {
}

ErrorCode Error::getCode() const {
  return _code;
}

const char *Error::what() const noexcept {
  return _what.c_str();
}

AbortedError::AbortedError(const std::string &what)
    : Error(ErrorCode::Aborted, buildErrorMsg(ErrorCode::Aborted, what)) {
}

OutOfRangeError::OutOfRangeError(const std::string &what)
    : Error(ErrorCode::OutOfRange, buildErrorMsg(ErrorCode::OutOfRange, what)) {
}

InvalidArgError::InvalidArgError(const std::string &what)
    : Error(ErrorCode::InvalidArg, buildErrorMsg(ErrorCode::InvalidArg, what)) {
}

NotImplementedError::NotImplementedError(const std::string &what)
    : Error(ErrorCode::InvalidArg, buildErrorMsg(ErrorCode::NotImplemented, what)) {
}

}  // namespace lut
