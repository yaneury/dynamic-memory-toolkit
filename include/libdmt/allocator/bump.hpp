#pragma once

#include <array>
#include <cstdlib>
#include <libdmt/internal/platform.hpp>
#include <libdmt/internal/types.hpp>
#include <libdmt/internal/util.hpp>

namespace dmt::allocator {

// Default size for the storage backed by the Bump allocator.
static constexpr std::size_t kDefaultSize = 4096;

struct SizeId {};

template <std::size_t Size>
struct SizeT : std::integral_constant<std::size_t, Size> {
  using Id_ = struct {};
};

template <std::size_t Alignment>
struct AlignmentT : std::integral_constant<std::size_t, Alignment> {
  using Id_ = struct {};
};

// TODO: Arena allocations when at capacity and using heap
template <class T, typename... Args> class Bump {
public:
  // Require alias for std::allocator_traits to infer other types, e.g.
  // using pointer = value_type*.
  using value_type = T;

  explicit Bump(){};

  ~Bump() { Reset(); }

  template <class U> constexpr Bump(const Bump<U>&) noexcept {}

  T* allocate(std::size_t n) noexcept {
    if (n > AlignedSize_)
      return nullptr;

    size_t request_size = internal::AlignUp(n, Alignment_);
    size_t remaining_size = AlignedSize_ - offset_;

    if (request_size > remaining_size)
      return nullptr;

    if (!chunk_.has_value()) {
      chunk_ = internal::AllocateBytes(AlignedSize_, Alignment_);
      if (!chunk_.has_value())
        return nullptr;
    }

    Byte* result = chunk_->base + offset_;
    offset_ += request_size;

    return reinterpret_cast<T*>(result);
  }

  void deallocate(T*, std::size_t) noexcept {
    // The bump allocator does not support per-object deallocation.
  }

  void Reset() {
    offset_ = 0;
    if (chunk_.has_value())
      internal::ReleaseBytes(chunk_.value());
    chunk_ = std::nullopt;
  }

private:
  using Byte = uint8_t;

  // There are several factors used to determine the alignment for the
  // allocator. First, users can specify their own alignment if desired using
  // |AlignmentT<>|. Otherwise, we use the alignment as determined by the C++
  // compiler. There's a floor in the size of the alignment to be equal to or
  // greater than |sizeof(void*)| for compatibility with std::aligned_alloc.
  static constexpr std::size_t Alignment_ =
      std::max({std::alignment_of_v<T>, sizeof(void*),
                internal::GetValueT<AlignmentT<0>, Args...>::value});

  static_assert(internal::IsPowerOfTwo(Alignment_),
                "Alignment must be a power of 2.");

  static constexpr std::size_t AlignedSize_ = internal::AlignUp(
      internal::GetValueT<SizeT<kDefaultSize>, Args...>::value, Alignment_);

  size_t offset_ = 0;
  std::optional<internal::Allocation> chunk_ = std::nullopt;
};

template <class T, class U> bool operator==(const Bump<T>&, const Bump<U>&) {
  return true;
}

template <class T, class U> bool operator!=(const Bump<T>&, const Bump<U>&) {
  return false;
}

} // namespace dmt::allocator