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

// Created at 2017-4-19

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include "lyutil/path.h"
#include "lyutil/reader.h"

namespace lut {

class IniSection;

// IniConfig is a class to read ini configuration file.
class IniConfig {
 public:
  // Read configuration from filename.
  static std::unique_ptr<IniConfig> read(const std::string &filename);

  // get section by name.
  const IniSection &getSection(const std::string &name) const;

  // returns true if section presents in the ini config.
  bool hasSection(const std::string &section) const;

  // Get filename.
  const std::string &getFilename() const { return _filename; }

 private:  
  std::string _filename;

  // map (section, key) -> value
  std::map<std::string, IniSection> _table;
  
  IniConfig();

  static bool isEmptyLine(const std::string &s);
  static bool isHeader(const std::string &s);
  static std::string parseHeader(const std::string &s);
  static std::pair<std::string, std::string> parseKeyValue(const std::string &s);
};

// one section in ini config.
class IniSection {
 public:
  friend class IniConfig;

  // get a value by section and key.
  std::string getString(const std::string &key) const;
  int getInt(const std::string &key) const;
  bool getBool(const std::string &key) const;
  float getFloat(const std::string &key) const;
  Path getPath(const std::string &key) const;

  // returns true if the key exists.
  bool hasKey(const std::string &key);

  // name of the section.
  const std::string &getName() const { return _name; }

 private:
  std::unordered_map<std::string, std::string> _kvTable;
  std::string _name;
  Path _iniDir;

  IniSection(const Path &iniDir);
};

} // namespace lut
