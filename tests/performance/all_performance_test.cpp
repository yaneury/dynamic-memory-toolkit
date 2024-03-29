#include <stack>

#include "catch2/catch_all.hpp"

#include <allocators/provider/lock_free_page.hpp>
#include <allocators/strategy/freelist.hpp>
#include <allocators/strategy/lock_free_bump.hpp>

#include "../util.hpp"

using namespace allocators;

template <class... Allocator> struct AllocatorPack {};

using AllocatorsUnderTest =
    AllocatorPack<strategy::LockFreeBump<provider::LockFreePage<>>,
                  strategy::FreeList<provider::LockFreePage<>>>;

TEMPLATE_LIST_TEST_CASE("Default allocators", "[allocator][all][performance]",
                        AllocatorsUnderTest) {
  // TODO: Enable once fixed.
  SKIP();
  // Use page-sized blocks for every allocator.
  static constexpr std::size_t kBlockSize = 4096;
  static constexpr std::array kRequestSizes = {
      1ul, 2ul, 4ul, 8ul, 16ul, 32ul, 64ul, 128ul, 256ul, 1024ul, 2048ul};

  using Allocator = TestType;

  auto make_allocations = []() {
    auto page_provider = provider::LockFreePage();
    Allocator allocator(page_provider);
    std::stack<std::byte*> allocations;
    for (std::size_t size : kRequestSizes) {
      auto p_or = allocator.Find(size);
      allocations.push(GetValueOrFail<std::byte*>(p_or));
    }

    if constexpr (std::is_same_v<Allocator, strategy::LockFreeBump<
                                                provider::LockFreePage<>>>) {
      REQUIRE(allocator.Reset().has_value());
    } else {
      while (allocations.size()) {
        INFO("Top: " << allocations.top());
        auto result = allocator.Return(allocations.top());
        INFO("Received error: " << (int)result.error());
        REQUIRE(result.has_value());
        allocations.pop();
      }
    }
  };

  BENCHMARK("Allocate and Release variable-sized objects on LIFO basis") {
    return make_allocations();
  };
}
