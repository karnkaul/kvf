#pragma once
#include "klib/base_types.hpp"
#include "kvf/buffer_write.hpp"
#include "kvf/render_device_fwd.hpp"
#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <gsl/pointers>
#include <memory>

namespace kvf {
enum class BufferType : std::uint8_t { Host, Device };

struct BufferCreateInfo {
	static constexpr vk::DeviceSize min_size_v{1};

	vk::BufferUsageFlags usage;

	BufferType type{BufferType::Host};
	vk::DeviceSize size{min_size_v};
};

class IBuffer : public klib::Polymorphic {
  public:
	using CreateInfo = BufferCreateInfo;

	[[nodiscard]] static auto create(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info) -> std::unique_ptr<IBuffer>;

	virtual void recreate(CreateInfo const& info) = 0;

	[[nodiscard]] virtual auto get_usage() const -> vk::BufferUsageFlags = 0;
	[[nodiscard]] virtual auto get_type() const -> BufferType = 0;
	[[nodiscard]] virtual auto get_buffer() const -> vk::Buffer = 0;
	[[nodiscard]] virtual auto get_mapped_ptr() const -> void* = 0;

	[[nodiscard]] virtual auto get_size() const -> vk::DeviceSize = 0;
	[[nodiscard]] virtual auto get_capacity() const -> vk::DeviceSize = 0;
	virtual void resize(vk::DeviceSize size) = 0;

	auto write_in_place(BufferWrite write, vk::DeviceSize offset = 0) -> bool;
	void resize_overwrite_contiguous(std::span<BufferWrite const> writes);
	void resize_and_overwrite(BufferWrite write) { resize_overwrite_contiguous({&write, 1}); }

	[[nodiscard]] auto get_mapped_span() const -> std::span<std::byte>;
	[[nodiscard]] auto descriptor_info() const -> vk::DescriptorBufferInfo;

  protected:
	virtual auto write_contiguous(std::span<BufferWrite const> writes, vk::DeviceSize write_size, vk::DeviceSize offset) -> bool = 0;
};
} // namespace kvf
