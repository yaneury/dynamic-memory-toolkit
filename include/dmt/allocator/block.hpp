#pragma once

#include <dmt/allocator/parameters.hpp>
#include <dmt/allocator/trait.hpp>
#include <dmt/internal/block.hpp>
#include <dmt/internal/util.hpp>
#include <template/parameters.hpp>

namespace dmt::allocator {

template <class... Args> class Block {
public:
  // Alignment used for the blocks requested. N.b. this is *not* the alignment
  // for individual allocation requests, of which may have different alignment
  // requirements.
  //
  // This field is optional. If not provided, will default to |sizeof(void*)|.
  // If provided, it must greater than |sizeof(void*)| and be a power of two.
  static constexpr std::size_t kAlignment =
      std::max({sizeof(void*), ntp::optional<AlignmentT<0>, Args...>::value});

  // Size of the blocks. This allocator doesn't support variable-sized blocks.
  // All blocks allocated are of the same size. N.b. that the size here will
  // *not* be the size of memory ultimately requested for blocks. This is so
  // because supplemental memory is needed for block headers and to ensure
  // alignment as specified with |kAlignment|.
  //
  // This field is optional. If not provided, will default to the common page
  // size, 4096.
  static constexpr std::size_t kSize =
      ntp::optional<SizeT<4096>, Args...>::value;

  // Sizing limits placed on |kSize|.
  // If |HaveAtLeastSizeBytes| is provided, then block must have |kSize| bytes
  // available not including header size and alignment.
  // If |NoMoreThanSizeBytes| is provided, then block must not exceed |kSize|
  // bytes, including after accounting for header size and alignment.
  static constexpr bool kMustContainSizeBytesInSpace =
      ntp::optional<LimitT<BlocksMust::HaveAtLeastSizeBytes>, Args...>::value ==
      BlocksMust::HaveAtLeastSizeBytes;

  // Policy employed when block has no more space for pending request.
  // If |GrowStorage| is provided, then a new block will be requested;
  // if |ReturnNull| is provided, then nullptr is returned on the allocation
  // request. This does not mean that it's impossible to request more memory
  // though. It only means that the block has no more space for the requested
  // size. If a smaller size request comes along, it may be possible that the
  // block has sufficient storage for it.
  static constexpr bool kGrowWhenFull =
      ntp::optional<GrowT<WhenFull::GrowStorage>, Args...>::value ==
      WhenFull::GrowStorage;

protected:
  // Ultimate size of the blocks after accounting for header and alignment.
  static constexpr std::size_t kAlignedSize_ =
      kMustContainSizeBytesInSpace
          ? internal::AlignUp(kSize + internal::GetBlockHeaderSize(),
                              kAlignment)
          : internal::AlignDown(kSize, kAlignment);

  static internal::Allocation CreateAllocation(std::byte* base) {
    std::size_t size = IsPageMultiple()
                           ? kAlignedSize_ / internal::GetPageSize()
                           : kAlignedSize_;
    return internal::Allocation{.base = static_cast<std::byte*>(base),
                                .size = size};
  }

  static bool IsPageMultiple() {
    static const auto page_size = internal::GetPageSize();
    return kAlignedSize_ >= page_size && kAlignedSize_ % page_size == 0;
  }

  static dmt::internal::BlockHeader* AllocateNewBlock() {
    auto allocation =
        IsPageMultiple()
            ? internal::AllocatePages(kAlignedSize_ / internal::GetPageSize())
            : internal::AllocateBytes(kAlignedSize_, kAlignment);

    if (!allocation.has_value())
      return nullptr;

    return dmt::internal::CreateBlockHeaderFromAllocation(allocation.value());
  }

  static void ReleaseBlocks(dmt::internal::BlockHeader* block) {
    auto release =
        IsPageMultiple() ? internal::ReleasePages : internal::ReleaseBytes;
    dmt::internal::ReleaseBlocks(block, std::move(release));
  }

  // Various assertions hidden from user API but added here to ensure invariants
  // are met at compile time.
  static_assert(internal::IsPowerOfTwo(kAlignment),
                "kAlignment must be a power of 2.");
};

} // namespace dmt::allocator
