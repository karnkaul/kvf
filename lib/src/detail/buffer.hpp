#pragma once
#include "kvf/buffer.hpp"
#include "kvf/render_device.hpp"

namespace kvf::detail {
class Buffer : public IBuffer {
  public:
	Buffer(Buffer const&) = delete;
	Buffer(Buffer&&) = delete;
	Buffer& operator=(Buffer const&) = delete;
	Buffer& operator=(Buffer&&) = delete;

	explicit Buffer(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info);
	~Buffer() { destroy(); }

  private:
	void recreate(CreateInfo const& create_info) final { recreate_impl(create_info); }

	[[nodiscard]] auto get_usage() const -> vk::BufferUsageFlags final { return m_info.usage; }
	[[nodiscard]] auto get_type() const -> BufferType final { return m_info.type; }
	[[nodiscard]] auto get_buffer() const -> vk::Buffer final { return m_buffer; }
	[[nodiscard]] auto get_mapped_ptr() const -> void* final { return m_mapped; }

	[[nodiscard]] auto get_size() const -> vk::DeviceSize final { return m_size; }
	[[nodiscard]] auto get_capacity() const -> vk::DeviceSize final { return m_info.size; }
	void resize(vk::DeviceSize size) final;

	auto write_contiguous(std::span<BufferWrite const> writes, vk::DeviceSize write_size, vk::DeviceSize offset) -> bool final;

	void recreate_impl(CreateInfo create_info);
	void destroy();

	gsl::not_null<IRenderDevice*> m_render_device;

	CreateInfo m_info{};
	vk::DeviceSize m_size{};

	VmaAllocation m_allocation{};
	vk::Buffer m_buffer{};
	void* m_mapped{};
};
} // namespace kvf::detail
