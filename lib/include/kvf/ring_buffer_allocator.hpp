#pragma once
#include "klib/base_types.hpp"
#include "kvf/fixed_usage_buffer.hpp"

namespace kvf {
class IRingBufferAllocator : public klib::Polymorphic {
  public:
	[[nodiscard]] virtual auto allocate_next() -> std::span<FixedUsageBuffer const> = 0;
};
} // namespace kvf
