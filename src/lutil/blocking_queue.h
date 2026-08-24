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

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

#include "lutil/noncopyable.h"

namespace lut {

/// @brief A FIFO queue that hands values from producer threads to consumer threads, blocking each
/// side when it has nothing to do. Any number of threads may use it at once and every method is
/// thread-safe.
///
/// The queue can be closed once, which is how producers tell consumers that no more values are
/// coming. Closing never discards what is already queued: waitPop() keeps handing out the
/// remaining values and only reports the end after the last one, so a consumer loop of
/// `while (queue.waitPop(value))` drains the queue and then exits on its own.
///
/// Giving it a maximum size makes push() wait while the queue is full, which stops a fast producer
/// from growing the queue without bound.
///
/// The queue must outlive every thread using it. Destroying it while a thread is blocked inside
/// push() or waitPop() is undefined behavior, so close the queue and join those threads first.
template<typename T>
class BlockingQueue : public NonCopyable {
 public:
  /// @brief Maximum size meaning the queue may grow as large as memory allows.
  static constexpr std::size_t kUnbounded = 0;

  /// @brief Construct a queue.
  /// @param maxSize how many values may be queued before push() starts waiting for room, or
  ///        kUnbounded to never wait.
  explicit BlockingQueue(std::size_t maxSize = kUnbounded)
      : _maxSize(maxSize) {
  }

  /// @brief Append a value, waiting while the queue is full. Returns immediately on an unbounded
  /// queue and on a closed one.
  /// @param value the value to append; moved into the queue on success.
  /// @return true on success, false if the queue is closed and the value was not queued.
  bool push(T value) {
    std::unique_lock<std::mutex> lock(_mutex);
    _notFull.wait(lock, [&]() {
      return _closed || _maxSize == kUnbounded || _queue.size() < _maxSize;
    });
    if (_closed) return false;

    _queue.push_back(std::move(value));
    lock.unlock();
    _notEmpty.notify_one();
    return true;
  }

  /// @brief Take the oldest value if one is queued, without ever waiting. Being closed makes no
  /// difference here: values queued before the close are still handed out.
  /// @param value assigned the value taken, untouched when there is none.
  /// @return true if a value was taken, false if the queue was empty.
  bool tryPop(T &value) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_queue.empty()) return false;

    value = std::move(_queue.front());
    _queue.pop_front();
    lock.unlock();
    _notFull.notify_one();
    return true;
  }

  /// @brief Take the oldest value, waiting for one to arrive if the queue is empty.
  /// @param value assigned the value taken, untouched when none is returned.
  /// @return true if a value was taken. False means the queue is closed and drained, and no
  ///         further value will ever arrive, which makes it a loop's exit condition.
  bool waitPop(T &value) {
    std::unique_lock<std::mutex> lock(_mutex);
    _notEmpty.wait(lock, [&]() { return _closed || !_queue.empty(); });
    if (_queue.empty()) return false;

    value = std::move(_queue.front());
    _queue.pop_front();
    lock.unlock();
    _notFull.notify_one();
    return true;
  }

  /// @brief Reject further pushes and wake every waiting thread. Values already queued stay
  /// available to the pop methods. Closing an already closed queue does nothing.
  void close() {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _closed = true;
    }
    _notEmpty.notify_all();
    _notFull.notify_all();
  }

  /// @brief Whether close() has been called. Once true it stays true.
  bool isClosed() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _closed;
  }

  /// @brief Whether the queue held no values at the moment of the call. Another thread may have
  /// changed that before the caller acts on the answer, so this cannot be used to decide that a
  /// later pop will succeed.
  bool empty() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _queue.empty();
  }

  /// @brief How many values the queue held at the moment of the call. Carries the same caveat as
  /// empty(): it is a snapshot, not a promise about what a later call will find.
  std::size_t size() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _queue.size();
  }

 private:
  mutable std::mutex _mutex;
  std::condition_variable _notEmpty;  ///< Waited on by the pop methods, signalled by push().
  std::condition_variable _notFull;   ///< Waited on by push(), signalled by the pop methods.
  std::deque<T> _queue;
  std::size_t _maxSize;
  bool _closed = false;
};

}  // namespace lut
