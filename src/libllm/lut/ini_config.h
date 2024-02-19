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
#include "libllm/lut/path.h"
#include "libllm/lut/reader.h"

namespace lut {

class IniSection;

// IniConfig is a class to read ini configuration file.
class IniConfig {
 public:
  // Read configuration from filename.
  static std::shared_ptr<IniConfig> fromFile(const std::string &filename);
  static std::shared_ptr<IniConfig> fromStream(lut::Reader *reader);

  // noncopyable.
  IniConfig(IniConfig &) = delete;
  IniConfig &operator=(IniConfig &) = delete;

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

  void readStream(lut::Reader *reader);
  bool isEmptyLine(const std::string &s);
  bool isHeader(const std::string &s);
  std::string parseHeader(const std::string &s);
  std::pair<std::string, std::string> parseKeyValue(const std::string &s);
};

// one section in ini config.
class IniSection {
 public:
  friend class IniConfig;

  std::string getString(const std::string &key) const;
  int getInt(const std::string &key) const;
  bool getBool(const std::string &key) const;
  float getFloat(const std::string &key) const;
  Path getPath(const std::string &key) const;

  /// @brief Get string list by key in section. Each element in the list is separate by ",".
  /// Speace will be kept between separators.
  /// @param key key of the string list field.
  /// @return the string list.
  std::vector<std::string> getStringArray(const std::string &key) const;

  /// @brief Get integer list by key in section. Each element in the list is separate by ",".
  /// @param key key of the integer list field.
  /// @return the integer list.
  std::vector<int> getIntArray(const std::string &key) const;

  // returns true if the key exists.
  bool hasKey(const std::string &key) const;

  // name of the section.
  const std::string &getName() const { return _name; }

 private:
  std::unordered_map<std::string, std::string> _kvTable;
  std::string _name;
  Path _iniDir;

  IniSection(const Path &iniDir);
};

} // namespace lut
