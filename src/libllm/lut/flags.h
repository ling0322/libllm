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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "libllm/lut/span.h"
#include "libllm/lut/strings.h"
#include "libllm/lut/log.h"

namespace lut {


// command line option parser.
// Example:
//   Flags flags;
//   std::string inputFile;
//   flags.define("-i", &inputFile, "the input file.");
//   flags.parse(argc, argv);
class Flags {
 public:
  class Parser;

  Flags(const std::string &usage);

  // define a string flag.
  void define(const std::string &flag, std::string *s, const std::string &usage = "");

  // parse argc and argv arguments.
  void parse(int argc, char *argv[]);

  // print usage.
  void printUsage() const;

  // get positional arguments after parsing.
  template<typename T>
  T getArg(int index);

 private:
  std::map<std::string, std::unique_ptr<Parser>> _parsers;
  std::vector<std::string> _positionalArgs;
  std::string _usage;
};

// interface for flags parser. It parse the argument given by Flags and store the parsed value into
// its bonded variable pointers.
class Flags::Parser {
 public:
  virtual ~Parser() = default;

  // parse the flag argument.
  virtual void parse(const std::string &arg) = 0;

  // gets the usage description.
  virtual std::string getUsage() const = 0;

  // gets the parameter type information in string.
  virtual std::string getType() const = 0;
};

} // namespace lut

