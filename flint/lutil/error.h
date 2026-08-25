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

#include <exception>
#include <string>

#define THROW(exType, msg)                                               \
  {                                                                      \
    std::string msg_ = msg;                                              \
    LOG(ERROR) << "an " << #exType << " exception was thrown: " << msg_; \
    throw lut::exType##Error(msg_);                                      \
  }

namespace lut {

enum class ErrorCode : int {
  OK = 0,
  Aborted = 1,
  OutOfRange = 2,
  InvalidArg = 3,
  NotImplemented = 4,
};

class Error : public std::exception {
 public:
  Error();
  Error(ErrorCode code, const std::string &what);
  ~Error();

  // get error code.
  ErrorCode getCode() const;

  // implement std::exception.
  const char *what() const noexcept override;

 private:
  ErrorCode _code;
  std::string _what;
};

class AbortedError : public Error {
 public:
  AbortedError(const std::string &what);
};

class OutOfRangeError : public Error {
 public:
  OutOfRangeError(const std::string &what);
};

class InvalidArgError : public Error {
 public:
  InvalidArgError(const std::string &what);
};

class NotImplementedError : public Error {
 public:
  NotImplementedError(const std::string &what);
};

}  // namespace lut
