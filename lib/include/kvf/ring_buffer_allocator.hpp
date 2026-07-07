#pragma once
#include "klib/base_types.hpp"
#include "kvf/fixed_usage_buffer.hpp"
#include <memory>

namespace kvf {
class IRingBufferAllocator : public klib::Polymorphic {
  public:
	[[nodiscard]] static auto create(gsl::not_null<IRenderDevice*> render_device, BufferUsageLayout const& layout) -> std::shared_ptr<IRingBufferAllocator>;

	[[nodiscard]] virtual auto get_render_device() const -> IRenderDevice& = 0;

	[[nodiscard]] virtual auto allocate_next() -> std::span<FixedUsageBuffer const> = 0;
};
} // namespace kvf
