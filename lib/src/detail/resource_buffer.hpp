#pragma once
#include "kvf/buffer.hpp"
#include "kvf/render_device.hpp"

namespace kvf::two::detail {
class ResourceBuffer : public IBuffer {
  public:
	ResourceBuffer(ResourceBuffer const&) = delete;
	ResourceBuffer(ResourceBuffer&&) = delete;
	ResourceBuffer& operator=(ResourceBuffer const&) = delete;
	ResourceBuffer& operator=(ResourceBuffer&&) = delete;

	explicit ResourceBuffer(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info);
	~ResourceBuffer() { destroy(); }

  private:
	void recreate(CreateInfo const& create_info) final { recreate_impl(create_info); }

	[[nodiscard]] auto get_usage() const -> vk::BufferUsageFlags final { return m_usage; }
	[[nodiscard]] auto get_type() const -> BufferType final { return m_type; }
	[[nodiscard]] auto get_buffer() const -> vk::Buffer final { return m_buffer; }
	[[nodiscard]] auto get_mapped_ptr() const -> void* final { return m_mapped; }

	[[nodiscard]] auto get_size() const -> vk::DeviceSize final { return m_size; }
	[[nodiscard]] auto get_capacity() const -> vk::DeviceSize final { return m_capacity; }
	void resize(vk::DeviceSize size) final;

	auto write_contiguous(std::span<BufferWrite const> writes, vk::DeviceSize write_size, vk::DeviceSize offset) -> bool final;

	void recreate_impl(CreateInfo const& create_info);
	void destroy();

	gsl::not_null<IRenderDevice*> m_render_device;

	vk::BufferUsageFlags m_usage{};
	BufferType m_type{};
	vk::DeviceSize m_capacity{};
	vk::DeviceSize m_size{};

	VmaAllocation m_allocation{};
	vk::Buffer m_buffer{};
	void* m_mapped{};
};
} // namespace kvf::two::detail
