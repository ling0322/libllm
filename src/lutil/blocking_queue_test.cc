#include "lutil/blocking_queue.h"

#include <chrono>
#include <future>
#include <string>
#include <thread>

#include "../../third_party/catch2/catch_amalgamated.hpp"

namespace lut {

CATCH_TEST_CASE("blocking queue preserves order and drains after close", "[core][util][queue]") {
  BlockingQueue<std::string> queue;
  CATCH_REQUIRE(queue.push("first"));
  CATCH_REQUIRE(queue.push("second"));
  CATCH_REQUIRE(queue.size() == 2);

  std::string value;
  CATCH_REQUIRE(queue.tryPop(value));
  CATCH_REQUIRE(value == "first");

  queue.close();
  CATCH_REQUIRE(queue.isClosed());
  CATCH_REQUIRE_FALSE(queue.push("third"));
  CATCH_REQUIRE(queue.waitPop(value));
  CATCH_REQUIRE(value == "second");
  CATCH_REQUIRE_FALSE(queue.waitPop(value));
}

CATCH_TEST_CASE("blocking queue wakes consumers", "[core][util][queue]") {
  BlockingQueue<int> queue;
  std::promise<int> receivedPromise;
  std::future<int> received = receivedPromise.get_future();
  std::thread consumer([&]() {
    int value = 0;
    if (queue.waitPop(value)) receivedPromise.set_value(value);
  });

  CATCH_REQUIRE(queue.push(42));
  CATCH_REQUIRE(received.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  CATCH_REQUIRE(received.get() == 42);
  consumer.join();
}

CATCH_TEST_CASE("closing blocking queue wakes an empty consumer", "[core][util][queue]") {
  BlockingQueue<int> queue;
  std::promise<bool> poppedPromise;
  std::future<bool> popped = poppedPromise.get_future();
  std::thread consumer([&]() {
    int value = 0;
    poppedPromise.set_value(queue.waitPop(value));
  });

  queue.close();
  CATCH_REQUIRE(popped.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  CATCH_REQUIRE_FALSE(popped.get());
  consumer.join();
}

CATCH_TEST_CASE("blocking queue empty() reflects contents", "[core][util][queue]") {
  BlockingQueue<int> queue;
  CATCH_REQUIRE(queue.empty());
  CATCH_REQUIRE(queue.push(1));
  CATCH_REQUIRE_FALSE(queue.empty());
}

CATCH_TEST_CASE("blocking queue applies back-pressure on a bounded push", "[core][util][queue]") {
  BlockingQueue<int> queue(1);
  CATCH_REQUIRE(queue.push(1));

  std::promise<bool> pushedPromise;
  std::future<bool> pushed = pushedPromise.get_future();
  std::thread producer([&]() { pushedPromise.set_value(queue.push(2)); });

  // capacity is 1 and already full, so the second push must block.
  CATCH_REQUIRE(pushed.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);

  int value = 0;
  CATCH_REQUIRE(queue.tryPop(value));
  CATCH_REQUIRE(value == 1);

  // popping frees a slot, which should wake the blocked producer.
  CATCH_REQUIRE(pushed.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  CATCH_REQUIRE(pushed.get());
  producer.join();

  CATCH_REQUIRE(queue.tryPop(value));
  CATCH_REQUIRE(value == 2);
}

CATCH_TEST_CASE("closing a full blocking queue wakes a blocked producer", "[core][util][queue]") {
  BlockingQueue<int> queue(1);
  CATCH_REQUIRE(queue.push(1));

  std::promise<bool> pushedPromise;
  std::future<bool> pushed = pushedPromise.get_future();
  std::thread producer([&]() { pushedPromise.set_value(queue.push(2)); });

  CATCH_REQUIRE(pushed.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);

  queue.close();

  CATCH_REQUIRE(pushed.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  CATCH_REQUIRE_FALSE(pushed.get());
  producer.join();
}

}  // namespace lut