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

#include "flint/memory.h"

#include "flint/operators.h"

namespace fl {

MemorySnapshot::MemorySnapshot(
    int64_t totalMemory,
    int64_t freeMemory,
    int64_t allocatedMemory,
    int64_t peakAllocatedMemory)
    : _totalMemory(totalMemory),
      _freeMemory(freeMemory),
      _allocatedMemory(allocatedMemory),
      _peakAllocatedMemory(peakAllocatedMemory) {
}

MemorySnapshot MemorySnapshot::capture(Device device) {
  return getOperators(device.getType())->captureMemorySnapshot();
}

void MemorySnapshot::resetPeakStats(Device device) {
  getOperators(device.getType())->resetPeakMemoryStats();
}

int64_t MemorySnapshot::getTotalMemory() const {
  return _totalMemory;
}

int64_t MemorySnapshot::getFreeMemory() const {
  return _freeMemory;
}

int64_t MemorySnapshot::getAllocatedMemory() const {
  return _allocatedMemory;
}

int64_t MemorySnapshot::getPeakAllocatedMemory() const {
  return _peakAllocatedMemory;
}

}  // namespace fl
