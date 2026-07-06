#pragma once
#include "klib/base_types.hpp"
#include "klib/enum/bitops.hpp"
#include "kvf/bitmap.hpp"
#include "kvf/render_target.hpp"
#include <cstdint>

namespace kvf {
enum class ImageFlag : std::int8_t {
	None = 0,
	DedicatedAlloc = 1 << 0,
	MipMapped = 1 << 1,
};
constexpr auto enable_enum_bitops(ImageFlag /*unused*/) { return true; }

struct ImageCreateInfo {
	static constexpr auto implicit_usage_v = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	static constexpr auto min_extent_v = vk::Extent2D{1, 1};

	vk::Format format;

	vk::ImageAspectFlags aspect{vk::ImageAspectFlagBits::eColor};
	vk::ImageUsageFlags usage{implicit_usage_v};
	vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
	std::uint32_t layers{1};
	vk::ImageViewType view_type{vk::ImageViewType::e2D};
	ImageFlag flags{};
	vk::Extent2D extent{min_extent_v};
};

class IImage : public klib::Polymorphic {
  public:
	using CreateInfo = ImageCreateInfo;

	virtual void recreate(CreateInfo const& info) = 0;

	[[nodiscard]] virtual auto get_format() const -> vk::Format = 0;
	[[nodiscard]] virtual auto get_aspect() const -> vk::ImageAspectFlags = 0;
	[[nodiscard]] virtual auto get_usage() const -> vk::ImageUsageFlags = 0;
	[[nodiscard]] virtual auto get_samples() const -> vk::SampleCountFlagBits = 0;
	[[nodiscard]] virtual auto get_layers() const -> std::uint32_t = 0;
	[[nodiscard]] virtual auto get_view_type() const -> vk::ImageViewType = 0;
	[[nodiscard]] virtual auto get_extent() const -> vk::Extent2D = 0;
	[[nodiscard]] virtual auto get_layout() const -> vk::ImageLayout = 0;
	[[nodiscard]] virtual auto get_image_view() const -> vk::ImageView = 0;
	[[nodiscard]] virtual auto get_image() const -> vk::Image = 0;
	[[nodiscard]] virtual auto get_mip_levels() const -> std::uint32_t = 0;

	virtual void resize(vk::Extent2D extent) = 0;
	virtual auto resize_and_overwrite(std::span<Bitmap const> layers) -> bool = 0;

	virtual void transition(vk::CommandBuffer command_buffer, vk::ImageMemoryBarrier2 barrier) = 0;

	auto resize_and_overwrite(Bitmap const& bitmap) -> bool { return resize_and_overwrite({&bitmap, 1}); }

	[[nodiscard]] auto subresource_range() const -> vk::ImageSubresourceRange;

	[[nodiscard]] auto render_target() const -> RenderTarget { return RenderTarget{.image = get_image(), .view = get_image_view(), .extent = get_extent()}; }

	[[nodiscard]] auto descriptor_info(vk::Sampler sampler) const -> vk::DescriptorImageInfo;
};
} // namespace kvf
