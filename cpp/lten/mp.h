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

#include <functional>

namespace lten {

// wrapper for OpenMP or other implementations
class MP {
 public:
  class Partition;
  class Context;

  static void init();
  static void destroy();
  static int getMaxThreads();

  /// @brief split range into N parts and apply each part in the closure. N is the number of
  /// workers in the thread pool.
  /// @param numBlocks number of blocks.
  /// @param closure the closure. Since we need to invoke the closure multiple times, we use it
  /// by value here.
  static void parallelFor(int numBlocks, std::function<void(Context)> closure);
};

/// @brief Store a partition info for a parallelFor function call.
class MP::Context {
 public:
  Context(int blockIdx, int numBlocks, int attachedThreadIdx)
      : _blockIdx(blockIdx),
        _numBlocks(numBlocks),
        _attachedThreadIdx(attachedThreadIdx) {
  }

  int getBlockIdx() const {
    return _blockIdx;
  }
  int getNumBlocks() const {
    return _numBlocks;
  }
  int getAttachedThreadIdx() const {
    return _attachedThreadIdx;
  }

 private:
  int _blockIdx;
  int _numBlocks;
  int _attachedThreadIdx;
};

}  // namespace lten
