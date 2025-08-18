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

#include "lutil/internal/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ctime>
#include <string>

#include "lutil/platform.h"

namespace lut {
namespace internal {

LogSeverity gLogLevel = LogSeverity::kINFO;

LogWrapper::LogWrapper(LogSeverity severity, const char *source_file, int source_line)
    : severity_(severity),
      source_line_(source_line) {
  const char *s = strrchr(source_file, '/');
  if (!s) {
    s = strrchr(source_file, '\\');
  }

  if (s) {
    source_file_ = s + 1;
  } else {
    source_file_ = s;
  }
}

LogWrapper::~LogWrapper() {
  std::string message = os_.str();
  if (message.empty()) message = default_message_;

  printf("%s %s %s:%d] %s\n", Severity(), Time(), source_file_, source_line_, message.c_str());

  if (severity_ == LogSeverity::kFATAL) {
    abort();
  }
}

const char *LogWrapper::Time() {
  time_t now = time(nullptr);

  std::strftime(time_, sizeof(time_), "%FT%TZ", std::gmtime(&now));
  return time_;
}

const char *LogWrapper::Severity() const {
  switch (severity_) {
    case LogSeverity::kDEBUG:
      return "DEBUG";
    case LogSeverity::kINFO:
      return "INFO";
    case LogSeverity::kWARN:
      return "WARNING";
    case LogSeverity::kERROR:
      return "ERROR";
    case LogSeverity::kFATAL:
      return "FATAL";
    default:
      fputs("invalid log severity.", stderr);
      abort();
  }
}

LogWrapper &LogWrapper::DefaultMessage(const char *message) {
  default_message_ = message;
  printStackTrace();
  return *this;
}

}  // namespace internal
}  // namespace lut

namespace lut {

void setLogLevel(LogSeverity level) {
  internal::gLogLevel = level;
}

}  // namespace lut
