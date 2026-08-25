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

#include "lutil/asio.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../third_party/catch2/catch_amalgamated.hpp"

namespace lut {

CATCH_TEST_CASE("event loop runs posted tasks in order", "[core][util][asio]") {
  EventLoop loop;
  std::vector<int> order;

  for (int i = 0; i < 4; ++i) {
    CATCH_REQUIRE(loop.post([&order, i]() { order.push_back(i); }));
  }
  loop.stop();
  loop.run();

  CATCH_REQUIRE(order == std::vector<int>{0, 1, 2, 3});
}

CATCH_TEST_CASE("event loop keeps running until stopped", "[core][util][asio]") {
  EventLoop loop;
  std::promise<void> ranPromise;
  std::future<void> ran = ranPromise.get_future();
  std::thread worker([&]() { loop.run(); });

  CATCH_REQUIRE(loop.post([&]() { ranPromise.set_value(); }));
  CATCH_REQUIRE(ran.wait_for(std::chrono::seconds(1)) == std::future_status::ready);

  // The task is done but run() has not returned, so the loop is still available.
  CATCH_REQUIRE_FALSE(loop.isStopped());
  loop.stop();
  worker.join();
  CATCH_REQUIRE(loop.isStopped());
}

CATCH_TEST_CASE("event loop rejects tasks once stopped", "[core][util][asio]") {
  EventLoop loop;
  loop.stop();

  bool ran = false;
  CATCH_REQUIRE_FALSE(loop.post([&ran]() { ran = true; }));
  loop.run();
  CATCH_REQUIRE_FALSE(ran);
}

CATCH_TEST_CASE("event loop poll runs ready tasks without blocking", "[core][util][asio]") {
  EventLoop loop;
  int counter = 0;

  // Nothing queued, so this must return at once rather than wait for work.
  CATCH_REQUIRE(loop.poll() == 0);

  CATCH_REQUIRE(loop.post([&counter]() { ++counter; }));
  CATCH_REQUIRE(loop.post([&counter]() { ++counter; }));
  CATCH_REQUIRE(loop.poll() == 2);
  CATCH_REQUIRE(counter == 2);
  CATCH_REQUIRE(loop.poll() == 0);
}

CATCH_TEST_CASE("event loop survives a throwing task", "[core][util][asio]") {
  EventLoop loop;
  bool ranAfterThrow = false;

  CATCH_REQUIRE(loop.post([]() { throw std::runtime_error("task failure"); }));
  CATCH_REQUIRE(loop.post([&ranAfterThrow]() { ranAfterThrow = true; }));
  loop.stop();
  loop.run();

  CATCH_REQUIRE(ranAfterThrow);
}

CATCH_TEST_CASE("event loop releases a task once it has run", "[core][util][asio]") {
  EventLoop loop;
  auto captured = std::make_shared<int>(1);
  std::weak_ptr<int> observer = captured;

  CATCH_REQUIRE(loop.post([captured]() {}));
  captured.reset();
  CATCH_REQUIRE_FALSE(observer.expired());

  CATCH_REQUIRE(loop.poll() == 1);
  CATCH_REQUIRE(observer.expired());
}

CATCH_TEST_CASE("event loop spreads tasks over several runners", "[core][util][asio]") {
  constexpr int NumThreads = 4;
  constexpr int NumTasks = 200;
  EventLoop loop;
  std::atomic<int> completed{0};
  std::vector<std::thread> runners;

  for (int i = 0; i < NumThreads; ++i) runners.emplace_back([&]() { loop.run(); });
  for (int i = 0; i < NumTasks; ++i) {
    CATCH_REQUIRE(loop.post([&completed]() { ++completed; }));
  }

  loop.stop();
  for (std::thread &runner : runners) runner.join();
  CATCH_REQUIRE(completed.load() == NumTasks);
}

}  // namespace lut
