#pragma once

#include <concepts> // TODO: Guard this against a C++20 check
#include <cstddef>

#include <dmt/internal/util.hpp>

namespace dmt::allocator {

// A parameter used for making an allocation request.
struct Layout {
  // Number of bytes requested for allocation.
  std::size_t size;

  // Alignment for allocated bytes.
  // Value must be a power of two and greater than or equal to
  // the current running architecture's word size, i.e. `sizeof(void*)`.
  std::size_t alignment;
};

template <class T>
concept Trait = requires(T allocator, std::size_t size, Layout layout,
                         std::byte* bytes) {
  { allocator.AllocateUnaligned(size) } -> std::same_as<std::byte*>;
  { allocator.Allocate(layout) } -> std::same_as<std::byte*>;
  { allocator.Release(bytes) } -> std::same_as<void>;
};

[[gnu::const]] inline bool IsValid(Layout layout) {
  return internal::IsValidRequest(layout.size, layout.alignment);
}

} // namespace dmt::allocator