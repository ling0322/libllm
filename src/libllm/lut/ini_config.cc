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

#include "libllm/lut/ini_config.h"

#include <tuple>
#include "libllm/lut/error.h"
#include "libllm/lut/log.h"
#include "libllm/lut/reader.h"
#include "libllm/lut/strings.h"

namespace lut {

// -- class IniConfig ----------------------------------------------------------

IniConfig::IniConfig() {}

std::unique_ptr<IniConfig> IniConfig::read(const std::string &filename) {
  std::unique_ptr<IniConfig> config{new IniConfig()};
  config->_filename = filename;

  auto fp = ReadableFile::open(filename);
  Scanner scanner{fp.get()};

  enum State {
    kBegin,
    kSelfLoop,
  };

  Path ini_dir = Path(filename).dirname();
  IniSection section{ini_dir};
  State state = kBegin;
  while (scanner.scan()) {
    std::string line = lut::trim(scanner.getText());
    if (state == kBegin) {
      if (isEmptyLine(line)) {
        // self-loop
      } else if (isHeader(line)) {
        section._name = parseHeader(line);
        state = kSelfLoop;
      } else {
        throw AbortedError(lut::sprintf("invalid line: %s", line));
      }
    } else if (state == kSelfLoop) {
      if (isEmptyLine(line)) {
        // do nothing.
      } else if (isHeader(line)) {
        config->_table.emplace(section.getName(), std::move(section));
        section = IniSection{ini_dir};
        section._name = parseHeader(line);
      } else {
        std::string key, value;
        std::tie(key, value) = parseKeyValue(line);
        section._kvTable[key] = value;
      }
    } else {
      NOT_IMPL();
    }
  }

  if (state == kBegin) {
    throw AbortedError("ini file is empty.");
  }

  config->_table.emplace(section.getName(), std::move(section));

  return config;
}

bool IniConfig::hasSection(const std::string &section) const {
  auto it = _table.find(section);
  return it != _table.end();
}


const IniSection &IniConfig::getSection(const std::string &name) const {
  auto it = _table.find(name);
  if (it == _table.end()) {
    throw lut::AbortedError(lut::sprintf("section not found: %s", name));
  }

  return it->second;
}

bool IniConfig::isEmptyLine(const std::string &s) {
  if (s.empty() || s.front() == ';') {
    return true;
  } else {
    return false;
  }
}

bool IniConfig::isHeader(const std::string &s) {
  if (s.front() == '[' && s.back() == ']') {
    return true;
  } else {
    return false;
  }
}

std::string IniConfig::parseHeader(const std::string &s) {
  if (!isHeader(s)) {
    throw AbortedError(lut::sprintf("invalid line: %s", s));
  }
  
  std::string name = s.substr(1, s.size() - 2);
  name = lut::trim(name);
  if (name.empty()) {
    throw AbortedError(lut::sprintf("invalid ini section: %s", s));
  }

  return name;
}

std::pair<std::string, std::string> IniConfig::parseKeyValue(const std::string &s) {
  auto row = lut::split(s, "=");
  if (row.size() != 2) {
    throw AbortedError(lut::sprintf("invalid line: %s", s));
  }
  std::string key = lut::toLower(lut::trim(row[0]));
  std::string value = lut::trim(row[1]);
  if (key.empty() || value.empty()) {
    throw AbortedError(lut::sprintf("invalid line: %s", s));
  }

  return std::make_pair(key, value);
}


// -- class IniSection ---------------------------------------------------------

IniSection::IniSection(const Path &iniDir) : _iniDir(iniDir) {}

std::string IniSection::getString(const std::string &key) const {
  auto it = _kvTable.find(key);
  if (it == _kvTable.end()) {
    THROW(Aborted, lut::sprintf("key not found (ini_session=%s): %s", _name, key));
  }

  return it->second;
}

int IniSection::getInt(const std::string &key) const {
  std::string s = getString(key);
  return lut::parseInt(s);
}

float IniSection::getFloat(const std::string &key) const {
  std::string s = getString(key);
  return lut::parseFloat(s);
}

bool IniSection::getBool(const std::string &key) const {
  return lut::parseBool(getString(key));
}

Path IniSection::getPath(const std::string &key) const {
  std::string s = getString(key);

  Path path(s);
  if (!path.isabs()) {
    return _iniDir / path;
  } else {
    return path;
  }
}

std::vector<std::string> IniSection::getStringArray(const std::string &key) const {
  std::string s = getString(key);
  std::vector<std::string> values = lut::split(s, ",");
  return values;
}

std::vector<int> IniSection::getIntArray(const std::string &key) const {
  std::string s = getString(key);
  std::vector<std::string> values = lut::split(s, ",");
  std::vector<int> intValues;

  for (const std::string &value : values) {
    intValues.push_back(lut::parseInt(value));
  }

  return intValues;
}

} // namespace lut
