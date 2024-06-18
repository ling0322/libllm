// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include <stdint.h>

#include <algorithm>

namespace lut {

class Range {
 public:
  class iterator {
   public:
    iterator(int begin, int end, int step)
        : _current(begin),
          _end(end),
          _step(step) {
    }

    int operator*() const {
      return _current;
    }

    iterator& operator++() {
      _current += _step;
      return *this;
    }

    bool operator!=(const iterator& other) const {
      // two iterator equals only when both are finished.
      return !((_current >= _end) && (other._current >= other._end));
    }

   private:
    int _current;
    int _step;
    int _end;
  };

  Range(int end)
      : _begin(0),
        _end(end),
        _step(1) {
  }
  Range(int begin, int end)
      : _begin(begin),
        _end(end),
        _step(1) {
  }
  Range(int begin, int end, int step)
      : _begin(begin),
        _end(end),
        _step(step) {
  }

  iterator begin() const {
    return iterator(_begin, _end, _step);
  }

  iterator end() const {
    return iterator(_end, _end, _step);
  }

  int getBegin() const {
    return _begin;
  }
  int getEnd() const {
    return _end;
  }
  int getStep() const {
    return _step;
  }

 private:
  int _begin;
  int _end;
  int _step;
};

}  // namespace lut
