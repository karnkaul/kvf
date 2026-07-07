#include "detail/ring_buffer_allocator.hpp"
#include "klib/debug/assert.hpp"
#include "kvf/render_device.hpp"
#include "log.hpp"

namespace kvf::detail {
namespace {
[[nodiscard]] constexpr auto grow_capacity(std::size_t const current, float const factor = 1.5f) {
	auto const fcap = float(std::max(current, 1uz));
	auto capacity = std::size_t(factor * fcap);
	if (capacity <= current) {
		++capacity;
	} else if (capacity > 8192) {
		capacity = current + 1;
	}
	return capacity;
}
} // namespace

RingBufferAllocator::RingBufferAllocator(gsl::not_null<IRenderDevice*> render_device, BufferUsageLayout const& usage_layout)
	: m_render_device(render_device), m_usage_layout(usage_layout) {
	KLIB_ASSERT(!m_usage_layout.empty());
}

auto RingBufferAllocator::allocate_next() -> std::span<FixedUsageBuffer const> {
	auto& pool = m_pools.at(std::size_t(m_frame_index));
	if (pool.index > 4096) { log.warn("{} Buffers allocated this frame", pool.index); }

	if (pool.index >= pool.layouts.size()) {
		pool.index = pool.layouts.size();
		auto const new_capacity = grow_capacity(pool.layouts.size());
		pool.layouts.reserve(new_capacity);
		while (pool.layouts.size() < new_capacity) {
			auto layout = BufferLayout{};
			layout.reserve(m_usage_layout.size());
			for (auto const usage : m_usage_layout) { layout.emplace_back(m_render_device, usage, BufferType::Host); }
			pool.layouts.push_back(std::move(layout));
		}
	}
	return pool.layouts.at(pool.index++);
}

void RingBufferAllocator::on_next_frame(FrameIndex const frame_index) {
	m_pools.at(std::size_t(frame_index)).index = 0;
	m_frame_index = frame_index;
}
} // namespace kvf::detail

namespace kvf {
auto IRingBufferAllocator::create(gsl::not_null<IRenderDevice*> render_device, BufferUsageLayout const& usage_layout) -> std::shared_ptr<IRingBufferAllocator> {
	auto ret = std::make_shared<detail::RingBufferAllocator>(render_device, usage_layout);
	render_device->attach_next_frame_listener(ret);
	return ret;
}
} // namespace kvf
