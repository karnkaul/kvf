#include "kvf/fixed_usage_buffer.hpp"
#include "klib/debug/assert.hpp"
#include "kvf/render_device.hpp"

namespace kvf {
FixedUsageBuffer::FixedUsageBuffer(gsl::not_null<IRenderDevice*> render_device, vk::BufferUsageFlags const usage, BufferType const type)
	: m_usage(usage), m_buffer(render_device->create_buffer(BufferCreateInfo{.usage = usage, .type = type})) {}

void FixedUsageBuffer::write(BufferWrite buffer_write) const {
	KLIB_ASSERT(m_buffer);
	m_buffer->resize_and_overwrite(buffer_write);
}

void FixedUsageBuffer::write_contiguous(std::span<BufferWrite const> buffer_writes) const {
	KLIB_ASSERT(m_buffer);
	m_buffer->resize_overwrite_contiguous(buffer_writes);
}

auto FixedUsageBuffer::get_buffer() const -> vk::Buffer {
	KLIB_ASSERT(m_buffer);
	return m_buffer->get_buffer();
}

auto FixedUsageBuffer::descriptor_info() const -> vk::DescriptorBufferInfo {
	KLIB_ASSERT(m_buffer);
	return m_buffer->descriptor_info();
}
} // namespace kvf
