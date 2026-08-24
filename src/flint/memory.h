// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "flint/device.h"

namespace fl {

/// @brief A snapshot of the memory usage of one device. Devices that do not report their memory
/// usage (the CPU backend) report zero for every value.
class MemorySnapshot {
 public:
  /// @brief Capture the current memory usage of a device.
  /// @param device The device to measure.
  /// @return The captured snapshot.
  static MemorySnapshot capture(Device device);

  /// @brief Set the peak allocated bytes of a device back to zero.
  /// @param device The device to reset.
  static void resetPeakStats(Device device);

  MemorySnapshot(
      int64_t totalMemory,
      int64_t freeMemory,
      int64_t allocatedMemory,
      int64_t peakAllocatedMemory);

  /// @brief Get the total memory of the device.
  /// @return The total memory in bytes.
  int64_t getTotalMemory() const;

  /// @brief Get the memory of the device that no process reserved yet. Memory that this process
  /// already took from the driver is not free even once its tensors are destroyed, because the
  /// allocator keeps it for reuse.
  /// @return The free memory in bytes.
  int64_t getFreeMemory() const;

  /// @brief Get the bytes that the tensors of this process currently hold.
  /// @return The allocated memory in bytes.
  int64_t getAllocatedMemory() const;

  /// @brief Get the largest value getAllocatedMemory() reached since the last resetPeakStats().
  /// @return The peak allocated memory in bytes.
  int64_t getPeakAllocatedMemory() const;

 private:
  int64_t _totalMemory;
  int64_t _freeMemory;
  int64_t _allocatedMemory;
  int64_t _peakAllocatedMemory;
};

}  // namespace fl
