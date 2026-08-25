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

#include <cstddef>
#include <exception>
#include <functional>
#include <utility>

#include "lutil/blocking_queue.h"
#include "lutil/log.h"
#include "lutil/noncopyable.h"

namespace lut {

/// @brief Runs work that other threads hand to it, one task at a time, in the order it arrives.
///
/// A thread that calls run() gives itself to the loop until the loop is stopped. Any other thread
/// can then hand it work with post(), which returns as soon as the task is queued. That is the
/// whole idea: state owned by the loop's thread is only ever touched from inside a task, so it
/// needs no locking of its own.
///
/// Several threads may call run() on the same loop, which turns it into a thread pool. Tasks are
/// then distributed among them and stop being ordered with respect to each other, so a loop whose
/// tasks share state should be run by one thread.
///
/// stop() lets the queued work finish rather than abandoning it: run() keeps taking tasks until
/// the queue is drained, and only then returns. A loop cannot be restarted afterwards.
///
/// A task that throws is reported and swallowed. Letting it escape run() would end the thread that
/// called run(), which usually means ending the process.
///
/// The loop must outlive every thread running or posting to it.
class EventLoop : public NonCopyable {
 public:
  using Task = std::function<void()>;

  /// @brief Maximum pending task count meaning the queue may grow as large as memory allows.
  static constexpr std::size_t kUnbounded = BlockingQueue<Task>::kUnbounded;

  /// @brief Construct a loop.
  /// @param maxPendingTasks how many tasks may be queued before post() starts waiting for the loop
  ///        to catch up, or kUnbounded to let the queue grow freely. Bounding it is only safe when
  ///        no task can block on the thread that posts, which would deadlock the two together.
  explicit EventLoop(std::size_t maxPendingTasks = kUnbounded)
      : _tasks(maxPendingTasks) {
  }

  /// @brief Hand a task to the loop and return without waiting for it to run.
  /// @param task the task to run; moved into the queue on success.
  /// @return true once queued, false if the loop is stopped and the task will never run.
  bool post(Task task) {
    return _tasks.push(std::move(task));
  }

  /// @brief Run tasks until the loop is stopped and the queued ones are done. Returns immediately
  /// on a loop that is already stopped and drained.
  void run() {
    Task task;
    while (_tasks.waitPop(task)) invoke(std::move(task));
  }

  /// @brief Run the tasks that are already queued, then return rather than waiting for more. Use
  /// it from a thread that has its own work to get back to.
  /// @return how many tasks ran.
  int poll() {
    int numTasks = 0;
    Task task;
    while (_tasks.tryPop(task)) {
      invoke(std::move(task));
      ++numTasks;
    }
    return numTasks;
  }

  /// @brief Stop accepting tasks and let run() return once the queued ones have run. Stopping a
  /// loop that is already stopped does nothing.
  void stop() {
    _tasks.close();
  }

  /// @brief Whether stop() has been called. Once true it stays true, though run() may still be
  /// working through what was already queued.
  bool isStopped() const {
    return _tasks.isClosed();
  }

 private:
  BlockingQueue<Task> _tasks;

  /// Taken by value so the task, and anything it captured, is released as soon as it has run
  /// rather than being held until the next one arrives.
  static void invoke(Task task) {
    try {
      task();
    } catch (const std::exception &e) {
      LOG(ERROR) << "event loop task failed: " << e.what();
    } catch (...) {
      LOG(ERROR) << "event loop task failed with an unknown exception";
    }
  }
};

}  // namespace lut
