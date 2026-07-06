#pragma once
#include "kvf/buffered.hpp"
#include "kvf/frame_index.hpp"
#include "kvf/render_device_fwd.hpp"
#include "kvf/vma.hpp"

namespace kvf {
class ScratchBuffer {
  public:
	class Allocator;

	explicit ScratchBuffer(gsl::not_null<IRenderApi const*> api, vk::BufferUsageFlags usage);

	[[nodiscard]] auto get_usage() const -> vk::BufferUsageFlags { return m_usage; }
	auto write(BufferWrite buffer_write) -> bool { return m_buffer.overwrite(buffer_write); }
	auto write_contiguous(std::span<BufferWrite const> buffer_writes) -> bool { return m_buffer.overwrite_contiguous(buffer_writes); }

	[[nodiscard]] auto get_buffer() const -> vk::Buffer { return m_buffer.get_buffer(); }
	[[nodiscard]] auto descriptor_info() const -> vk::DescriptorBufferInfo { return m_buffer.descriptor_info(); }

  private:
	vk::BufferUsageFlags m_usage{};
	vma::Buffer m_buffer{};
};

class ScratchBuffer::Allocator {
  public:
	using UsageLayout = std::vector<vk::BufferUsageFlags>;

	explicit Allocator(gsl::not_null<RenderDevice const*> render_device, UsageLayout usage_layout);

	void next_frame();

	[[nodiscard]] auto allocate_next() -> std::span<ScratchBuffer>;

  private:
	using BufferLayout = std::vector<ScratchBuffer>;

	struct Pool {
		std::vector<BufferLayout> layouts{};
		std::size_t index{};
	};

	gsl::not_null<RenderDevice const*> m_render_device;
	UsageLayout m_usage_layout{};

	FrameIndex m_frame_index{};
	Buffered<Pool> m_pools{};
};
} // namespace kvf
