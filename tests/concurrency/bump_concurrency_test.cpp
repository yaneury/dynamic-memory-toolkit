#include "catch2/catch_all.hpp"

#include <mutex>
#include <thread>
#include <vector>

#include <allocators/provider/lock_free_page.hpp>
#include <allocators/strategy/lock_free_bump.hpp>

#include "../util.hpp"

using namespace allocators;

using AllocatorUnderTest = strategy::LockFreeBump<provider::LockFreePage<>>;

TEST_CASE("LockFreeBump allocator works in multi-threaded contexts",
          "[concurrency][allocator][LockFreeBump]") {
  static constexpr std::size_t kNumThreads = 64;

  provider::LockFreePage<> provider;
  AllocatorUnderTest allocator(provider);
  std::mutex catch_mutex;

  auto allocate = [&]() {
    std::size_t count = GetRandomNumber(1, 100);
    {
      std::scoped_lock lock(catch_mutex);
      INFO("[" << std::this_thread::get_id() << "]: Thread will create "
               << count << "allocations.");
    }
    for (auto i = 0ul; i < count; ++i) {
      auto p_or = allocator.Find(GetRandomNumber(1, 100));
      REQUIRE(p_or.has_value());
    }
  };

  allocate();

  std::vector<std::thread> threads;
  for (auto i = 0ul; i < kNumThreads; ++i)
    threads.emplace_back(allocate);

  for (auto& th : threads)
    th.join();

  auto result = allocator.Reset();
  if (result.has_error())
    INFO("Reset call failed with: " << ToString(result.error()));
  REQUIRE(result.has_value());
}
