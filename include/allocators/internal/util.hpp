#pragma once

#include <concepts>

#ifdef DEBUG

#include <plog/Log.h>

#define DINFO(x) PLOGD << x;
#define DERROR(x) PLOGE << x;
#else
#define DINFO(x)
#define DERROR(x)
#endif

#define ALLOCATORS_NO_COPY_NO_MOVE_NO_DEFAULT(class_)                          \
  class_() = delete;                                                           \
  class_(const class_&) = delete;                                              \
  class_& operator=(const class_&) = delete;                                   \
  class_(class_&&) = delete;                                                   \
  class_& operator=(class_&&) = delete;

#define ALLOCATORS_NO_COPY_NO_MOVE(class_)                                     \
  class_(const class_&) = delete;                                              \
  class_& operator=(const class_&) = delete;                                   \
  class_(class_&&) = delete;                                                   \
  class_& operator=(class_&&) = delete;

#include <cstddef>

namespace allocators::internal {

static constexpr size_t kMinimumAlignment = sizeof(void*);

[[gnu::const]] inline constexpr bool IsPowerOfTwo(std::size_t n) {
  return n && !(n & (n - 1));
}

[[gnu::const]] inline constexpr std::size_t AlignUp(std::size_t n,
                                                    std::size_t alignment) {
  if (!n || !alignment)
    return 0;

  return (n + alignment - 1) & ~(alignment - 1);
}

[[gnu::const]] inline constexpr std::size_t AlignDown(std::size_t n,
                                                      std::size_t alignment) {
  if (!n || !alignment)
    return 0;

  return (n & ~(alignment - 1));
}

[[gnu::const]] inline constexpr bool IsValidAlignment(std::size_t alignment) {
  return alignment >= kMinimumAlignment && IsPowerOfTwo(alignment);
}

[[gnu::const]] inline bool IsValidRequest(std::size_t size,
                                          std::size_t alignment) {
  return size && IsValidAlignment(alignment);
}

template <class T> T* PtrAdd(T* ptr, std::size_t offset) {
  auto addr = reinterpret_cast<std::size_t>(ptr);
  return reinterpret_cast<T*>(addr + offset);
}

template <class T> T* PtrSubtract(T* ptr, std::size_t offset) {
  auto addr = reinterpret_cast<std::size_t>(ptr);
  return reinterpret_cast<T*>(addr - offset);
}

template <std::integral T> std::byte* ToBytePtr(T ptr) {
  return reinterpret_cast<std::byte*>(ptr);
}

template <std::integral T> T FromBytePtr(std::byte* ptr) {
  return reinterpret_cast<T>(ptr);
}

} // namespace allocators::internal
