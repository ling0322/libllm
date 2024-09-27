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

#include "lutil/ini_config.h"

#include <tuple>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/reader.h"
#include "lutil/strings.h"

namespace lut {

// -- class IniConfig ----------------------------------------------------------

IniConfig::IniConfig() {
}

std::shared_ptr<IniConfig> IniConfig::fromFile(const std::string &filename) {
  std::shared_ptr<IniConfig> config{new IniConfig()};
  config->_filename = filename;

  std::shared_ptr<Reader> reader = lut::ReadableFile::open(filename);
  config->readStream(reader.get());

  return config;
}

std::shared_ptr<IniConfig> IniConfig::fromStream(lut::Reader *reader) {
  std::shared_ptr<IniConfig> config{new IniConfig()};
  config->readStream(reader);

  return config;
}

void IniConfig::readStream(lut::Reader *fp) {
  _table.clear();

  enum State {
    kBegin,
    kSelfLoop,
  };

  Path iniDir = Path(_filename).dirname();
  IniSection section{iniDir};
  State state = kBegin;
  std::string line;
  while ((line = fp->readLine()) != "") {
    line = lut::trim(line);
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
        _table.emplace(section.getName(), std::move(section));
        section = IniSection{iniDir};
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

  _table.emplace(section.getName(), std::move(section));
}

bool IniConfig::hasSection(const std::string &section) const {
  auto it = _table.find(section);
  return it != _table.end();
}

const IniSection &IniConfig::getSection(const std::string &name) const {
  auto it = _table.find(name);
  if (it == _table.end()) {
    THROW(Aborted, lut::sprintf("section not found: %s", name));
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

IniSection::IniSection(const Path &iniDir)
    : _iniDir(iniDir) {
}

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

bool IniSection::hasKey(const std::string &key) const {
  return _kvTable.find(key) != _kvTable.end();
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

}  // namespace lut
