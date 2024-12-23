#pragma once
#include <vk_mem_alloc.h>
#include <klib/unique.hpp>
#include <kvf/render_device_fwd.hpp>
#include <kvf/render_target.hpp>
#include <gsl/pointers>

namespace kvf::vma {
template <typename Type>
struct Resource {
	VmaAllocator allocator{};
	VmaAllocation allocation{};
	Type resource{};

	auto operator==(Resource const& rhs) const -> bool { return allocation == rhs.allocation; }
};

struct ImageCreateInfo {
	vk::Extent2D extent;
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

	Image() = default;

	explicit Image(gsl::not_null<RenderDevice const*> render_device, CreateInfo const& create_info) : m_device(render_device) { recreate(create_info); }

	void recreate(CreateInfo const& create_info);

	[[nodiscard]] auto get_image() const -> vk::Image { return m_image.get().resource; }
	[[nodiscard]] auto get_view() const -> vk::ImageView { return *m_view; }
	[[nodiscard]] auto get_extent() const -> vk::Extent2D { return m_extent; }

	[[nodiscard]] auto render_target() const -> RenderTarget { return RenderTarget{.image = get_image(), .view = get_view(), .extent = get_extent()}; }

  private:
	struct Deleter {
		void operator()(Resource<vk::Image> const& image) const noexcept;
	};

	RenderDevice const* m_device{};
	klib::Unique<Resource<vk::Image>, Deleter> m_image{};
	vk::UniqueImageView m_view{};
	vk::Extent2D m_extent{};
};
} // namespace kvf::vma
