#pragma once
#include "kvf/next_frame_listener.hpp"
#include "kvf/ring.hpp"
#include "kvf/ring_buffer_allocator.hpp"

namespace kvf::detail {
class RingBufferAllocator : public IRingBufferAllocator, public INextFrameListener {
  public:
	explicit RingBufferAllocator(gsl::not_null<IRenderDevice*> render_device, BufferUsageLayout const& usage_layout);

  private:
	using BufferLayout = std::vector<FixedUsageBuffer>;

	struct Pool {
		std::vector<BufferLayout> layouts{};
		std::size_t index{};
	};

	[[nodiscard]] auto get_render_device() const -> IRenderDevice& final { return *m_render_device; }

	[[nodiscard]] auto allocate_next() -> std::span<FixedUsageBuffer const> final;

	void on_next_frame(FrameIndex frame_index) final;

	gsl::not_null<IRenderDevice*> m_render_device;
	BufferUsageLayout m_usage_layout{};

	FrameIndex m_frame_index{};
	Ring<Pool> m_pools{};
};
} // namespace kvf::detail
