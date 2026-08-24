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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "lutil/noncopyable.h"

namespace lut {

/// @brief A one-way flag that tells a long-running operation to stop, and that other threads can
/// wait on. Every method is thread-safe.
///
/// Cancellation only ever moves from off to on. That makes it safe to observe without further
/// synchronization: a caller that sees the token cancelled knows it will never be un-cancelled,
/// which is what lets isCancelled() be checked in a loop without a lock.
///
/// The token itself is not copyable, since a canceller and its observers have to share one flag.
/// Hand it out as a `std::shared_ptr<CancellationToken>`, or keep it in a longer-lived object and
/// pass a reference. Either way, it must outlive every thread that touches it.
///
/// There is no separate source type: only cancel() is non-const, so passing a worker a
/// `const CancellationToken &` already lets it observe and wait without being able to cancel,
/// which is the separation a CancellationTokenSource exists to provide elsewhere.
///
/// It carries no reason, deadline, or callbacks. A caller that needs to stop waiting on something
/// else when cancellation arrives still has to arrange for that itself, for example by closing the
/// queue it is blocked on.
class CancellationToken : public NonCopyable {
 public:
  CancellationToken() = default;

  /// @brief Request cancellation and wake everything waiting on this token. Cancelling a token
  /// that is already cancelled does nothing.
  void cancel() {
    {
      // Written under the lock so a waiter cannot evaluate its condition, find it false, and go
      // to sleep after this store but before the notify below.
      std::lock_guard<std::mutex> lock(_mutex);
      _cancelled.store(true, std::memory_order_release);
    }
    _cancelledCv.notify_all();
  }

  /// @brief Whether cancellation has been requested. Once true it stays true, so this is meant to
  /// be polled from inside a loop.
  /// @return true if cancel() has been called.
  bool isCancelled() const {
    return _cancelled.load(std::memory_order_acquire);
  }

  /// @brief Block until cancel() is called. Returns immediately if it already was.
  void wait() const {
    std::unique_lock<std::mutex> lock(_mutex);
    _cancelledCv.wait(lock, [this]() { return isCancelled(); });
  }

  /// @brief Block until cancel() is called or `timeout` elapses, whichever comes first.
  /// @param timeout how long to wait at most.
  /// @return true if the token was cancelled, false if the wait timed out first.
  template<typename Rep, typename Period>
  bool waitFor(const std::chrono::duration<Rep, Period> &timeout) const {
    std::unique_lock<std::mutex> lock(_mutex);
    return _cancelledCv.wait_for(lock, timeout, [this]() { return isCancelled(); });
  }

 private:
  mutable std::mutex _mutex;
  mutable std::condition_variable _cancelledCv;

  // Atomic so that isCancelled() stays cheap enough to poll, while cancel() still writes it under
  // _mutex to keep the condition variable protocol intact.
  std::atomic<bool> _cancelled{false};
};

}  // namespace lut
