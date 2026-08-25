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

#include "lutil/cancellation_token.h"

#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "../../third_party/catch2/catch_amalgamated.hpp"

namespace lut {

CATCH_TEST_CASE("cancellation token starts uncancelled and latches", "[core][util][cancel]") {
  CancellationToken token;
  CATCH_REQUIRE_FALSE(token.isCancelled());

  token.cancel();
  CATCH_REQUIRE(token.isCancelled());

  // Cancelling twice is a no-op rather than an error.
  token.cancel();
  CATCH_REQUIRE(token.isCancelled());
}

CATCH_TEST_CASE("cancellation token wakes a waiting thread", "[core][util][cancel]") {
  CancellationToken token;
  std::promise<void> wokePromise;
  std::future<void> woke = wokePromise.get_future();
  std::thread waiter([&]() {
    token.wait();
    wokePromise.set_value();
  });

  CATCH_REQUIRE(woke.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
  token.cancel();
  CATCH_REQUIRE(woke.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  waiter.join();
}

CATCH_TEST_CASE("waiting on an already cancelled token returns at once", "[core][util][cancel]") {
  CancellationToken token;
  token.cancel();

  token.wait();
  CATCH_REQUIRE(token.waitFor(std::chrono::seconds(0)));
}

CATCH_TEST_CASE("cancellation token waitFor reports a timeout", "[core][util][cancel]") {
  CancellationToken token;
  CATCH_REQUIRE_FALSE(token.waitFor(std::chrono::milliseconds(10)));
  CATCH_REQUIRE_FALSE(token.isCancelled());
}

CATCH_TEST_CASE("cancellation token releases every waiter", "[core][util][cancel]") {
  constexpr int NumWaiters = 8;
  CancellationToken token;
  std::vector<std::thread> waiters;
  std::atomic<int> woken{0};

  for (int i = 0; i < NumWaiters; ++i) {
    waiters.emplace_back([&]() {
      token.wait();
      ++woken;
    });
  }

  token.cancel();
  for (std::thread &waiter : waiters) waiter.join();
  CATCH_REQUIRE(woken.load() == NumWaiters);
}

}  // namespace lut
