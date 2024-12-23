#pragma once
#include <vk_mem_alloc.h>
#include <klib/unique.hpp>
#include <kvf/render_device_fwd.hpp>
#include <kvf/render_target.hpp>
#include <cstdint>
#include <gsl/pointers>

namespace kvf::vma {
template <typename Type>
concept ResourceT = std::same_as<Type, vk::Buffer> || std::same_as<Type, vk::Image>;

template <ResourceT Type>
struct Resource {
	VmaAllocator allocator{};
	VmaAllocation allocation{};
	Type resource{};

	auto operator==(Resource const& rhs) const -> bool { return allocation == rhs.allocation; }
};

enum class BufferType : std::uint8_t { Host, Device };

struct BufferCreateInfo {
	vk::BufferUsageFlags usage;

	BufferType type{BufferType::Host};
};

class Buffer {
  public:
	using CreateInfo = BufferCreateInfo;

	static constexpr auto min_size_v = vk::DeviceSize{1};

	Buffer() = default;

	explicit Buffer(gsl::not_null<RenderDevice const*> render_device, CreateInfo const& create_info, vk::DeviceSize size = min_size_v);

	auto resize(vk::DeviceSize size) -> bool;

	[[nodiscard]] auto get_buffer() const -> vk::Buffer { return m_buffer.get().resource; }
	[[nodiscard]] auto get_mapped() const -> void* { return m_mapped; }

	[[nodiscard]] auto get_capacity() const -> vk::DeviceSize { return m_capacity; }
	[[nodiscard]] auto get_size() const -> vk::DeviceSize { return m_size; }
	[[nodiscard]] auto get_info() const -> CreateInfo const& { return m_create_info; }

	explicit operator bool() const { return m_buffer.get().resource != vk::Buffer{}; }

  private:
	struct Deleter {
		void operator()(Resource<vk::Buffer> const& buffer) const noexcept;
	};

	RenderDevice const* m_device{};
	CreateInfo m_create_info{};
	klib::Unique<Resource<vk::Buffer>, Deleter> m_buffer{};
	vk::DeviceSize m_capacity{};
	vk::DeviceSize m_size{};
	void* m_mapped{};
};

struct ImageCreateInfo {
	vk::Format format;
	vk::ImageUsageFlags usage;
	vk::ImageAspectFlags aspect;

	vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
	std::uint32_t layers{1};
	std::uint32_t mips{1};
	vk::ImageViewType view_type{vk::ImageViewType::e2D};
};

class Image {
  public:
	using CreateInfo = ImageCreateInfo;

	static constexpr auto min_extent_v = vk::Extent2D{1, 1};

	Image() = default;

	explicit Image(gsl::not_null<RenderDevice const*> render_device, CreateInfo const& create_info, vk::Extent2D extent = min_extent_v);

	auto resize(vk::Extent2D extent) -> bool;

	[[nodiscard]] auto get_image() const -> vk::Image { return m_image.get().resource; }
	[[nodiscard]] auto get_view() const -> vk::ImageView { return *m_view; }

	[[nodiscard]] auto get_extent() const -> vk::Extent2D { return m_extent; }
	[[nodiscard]] auto get_info() const -> CreateInfo const& { return m_create_info; }

	[[nodiscard]] auto render_target() const -> RenderTarget { return RenderTarget{.image = get_image(), .view = get_view(), .extent = get_extent()}; }

	explicit operator bool() const { return m_image.get().resource != vk::Image{}; }

  private:
	struct Deleter {
		void operator()(Resource<vk::Image> const& image) const noexcept;
	};

	RenderDevice const* m_device{};
	CreateInfo m_create_info{};
	klib::Unique<Resource<vk::Image>, Deleter> m_image{};
	vk::UniqueImageView m_view{};
	vk::Extent2D m_extent{};
};
} // namespace kvf::vma
