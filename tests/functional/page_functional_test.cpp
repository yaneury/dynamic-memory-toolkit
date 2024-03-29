#include "catch2/catch_all.hpp"

#include <array>

#include <allocators/provider/lock_free_page.hpp>
#include <allocators/provider/unsynchronized_page.hpp>

using namespace allocators;

static constexpr std::size_t kPageSize = 4096;
static constexpr std::uint64_t kMaxPages = (1 << 18) - 1;

template <class... Allocator> struct AllocatorPack {};

using AllocatorsUnderTest =
    AllocatorPack<provider::LockFreePage<>, provider::UnsynchronizedPage<>>;

TEMPLATE_LIST_TEST_CASE("Page allocator", "[functional][allocator][Page]",
                        AllocatorsUnderTest) {
  using AllocatorUnderTest = TestType;

  SECTION("Can allocate 1 * kMaxPages worth of pages") {
    std::array<std::byte*, kMaxPages> allocations = {};
    AllocatorUnderTest allocator;

    for (auto i = 0u; i < kMaxPages; ++i) {
      auto p_or = allocator.Provide(1);
      REQUIRE(p_or.has_value());
      REQUIRE(p_or.value() != nullptr);
      allocations[i] = p_or.value();
    }

    for (auto i = 0u; i < kMaxPages; ++i) {
      REQUIRE_NOTHROW([&]() {
        std::byte* p = allocations[i];
        for (int j = 0; j < kPageSize; ++j)
          p[j] = std::byte();
      });
    }

    for (auto i = 0u; i < kMaxPages; ++i) {
      auto result = allocator.Return(allocations[i]);
      REQUIRE(result.has_value());
    }
  }

  // TODO: Support multiples pages per request.
  SECTION("Can allocator multiple pages per request") {}

  SECTION("While rejecting invalid sizes") {
    AllocatorUnderTest allocator;
    for (auto size : {0ul}) {
      auto p_or = allocator.Provide(size);
      REQUIRE(p_or.has_error());
      REQUIRE(p_or.error() == Error::InvalidInput);
    }
  }
}
