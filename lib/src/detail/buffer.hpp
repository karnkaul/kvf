#pragma once
#include "detail/vma.hpp"
#include "kvf/buffer.hpp"
#include "kvf/kvf_fwd.hpp"

namespace kvf::detail {
class Buffer : public IBuffer {
  public:
	explicit Buffer(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info);

  private:
	void recreate(CreateInfo const& create_info) final { recreate_impl(create_info); }

	[[nodiscard]] auto get_render_device() const -> IRenderDevice& final { return *m_render_device; }

	[[nodiscard]] auto get_usage() const -> vk::BufferUsageFlags final { return m_info.usage; }
	[[nodiscard]] auto get_type() const -> BufferType final { return m_info.type; }
	[[nodiscard]] auto get_buffer() const -> vk::Buffer final { return m_buffer.get().buffer; }
	[[nodiscard]] auto get_mapped_ptr() const -> void* final { return m_buffer.get().mapped; }

	[[nodiscard]] auto get_size() const -> vk::DeviceSize final { return m_size; }
	[[nodiscard]] auto get_capacity() const -> vk::DeviceSize final { return m_info.size; }
	void resize(vk::DeviceSize size) final;

	auto write_contiguous(std::span<BufferWrite const> writes, vk::DeviceSize write_size, vk::DeviceSize offset) -> bool final;

	void recreate_impl(CreateInfo create_info);

	gsl::not_null<IRenderDevice*> m_render_device;

	CreateInfo m_info{};
	vma::UniqueBuffer m_buffer{};

	vk::DeviceSize m_size{};
};
} // namespace kvf::detail
