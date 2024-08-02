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

#include "lut/range.h"

namespace libllm {

// wrapper for OpenMP or other implementations
class MP {
 public:
  class Partition;

  static void init();
  static void destroy();
  static int getMaxThreads();

  /// @brief split range into N parts and apply each part in the closure. N is the number of
  /// workers in the thread pool.
  /// @param range the range.
  /// @param closure the closure. Since we need to invoke the closure multiple times, we use it
  /// by value here.
  /// @param numThreads number of threads to use. -1 means using all threads in the pool.
  static void parallelFor(lut::Range range, int numThreads, std::function<void(Partition)> closure);
  static void parallelFor(lut::Range range, std::function<void(Partition)> closure);

 private:
  /// @brief Split the range into N parts, and returns the i-th part.
  /// @param range the range to split.
  /// @param chunkIdx the i-th part to get.
  /// @param numChunks number of parts to split (the N).
  /// @return i-th part of the range after split.
  static lut::Range splitRange(lut::Range range, int chunkIdx, int numChunks);
};

/// @brief Store a partition info for a parallelFor function call.
class MP::Partition {
 public:
  Partition(lut::Range range);
  Partition(lut::Range range, int partitionIdx, int numPartitions, int attachedThreadIdx);

  lut::Range getRange() const;
  int getPartitionIdx() const;
  int getNumPartitions() const;
  int getAttachedThreadIdx() const;

 private:
  lut::Range _range;
  int _partitionIdx;
  int _numPartitions;
  int _attachedThreadIdx;
};

}  // namespace libllm
