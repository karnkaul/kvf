#pragma once
#include "kvf/kvf_fwd.hpp"
#include "kvf/render_buffer.hpp"
#include <gsl/pointers>
#include <span>

namespace kvf {
using BufferUsageLayout = std::span<vk::BufferUsageFlags const>;

class FixedUsageBuffer {
  public:
	explicit FixedUsageBuffer(gsl::not_null<IRenderDevice*> device, vk::BufferUsageFlags usage, BufferType type = BufferType::Host);

	[[nodiscard]] auto get_render_device() const -> IRenderDevice& { return m_buffer->get_render_device(); }

	[[nodiscard]] auto get_usage() const -> vk::BufferUsageFlags { return m_usage; }
	[[nodiscard]] auto get_type() const -> BufferType { return m_type; }

	void write(BufferWrite buffer_write) const;
	void write_contiguous(std::span<BufferWrite const> buffer_writes) const;

	[[nodiscard]] auto get_buffer() const -> vk::Buffer;
	[[nodiscard]] auto descriptor_info() const -> vk::DescriptorBufferInfo;

  private:
	vk::BufferUsageFlags m_usage{};
	BufferType m_type{};
	std::unique_ptr<IRenderBuffer> m_buffer{};
};
} // namespace kvf
