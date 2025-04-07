#pragma once
#include <vk_mem_alloc.h>
#include <klib/base_types.hpp>
#include <klib/enum_flags.hpp>
#include <klib/unique.hpp>
#include <kvf/bitmap.hpp>
#include <kvf/buffer_write.hpp>
#include <kvf/render_api.hpp>
#include <kvf/render_target.hpp>
#include <kvf/vma_fwd.hpp>
#include <cstdint>
#include <gsl/pointers>

namespace kvf::vma {
template <typename Type>
concept ResourceT = std::same_as<Type, vk::Buffer> || std::same_as<Type, vk::Image>;

template <ResourceT Type>
class Resource : public klib::Polymorphic {
  public:
	Resource() = default;

	[[nodiscard]] auto get_render_api() const -> IRenderApi const* { return m_api; }

	explicit operator bool() const { return m_api != nullptr; }

  protected:
	explicit Resource(gsl::not_null<IRenderApi const*> api) : m_api(api) {}

	struct Payload {
		VmaAllocator allocator{};
		VmaAllocation allocation{};
		Type resource{};

		auto operator==(Payload const& rhs) const -> bool { return allocation == rhs.allocation; }
	};

	IRenderApi const* m_api{};
};

enum class BufferType : std::uint8_t { Host, Device };

struct BufferCreateInfo {
	vk::BufferUsageFlags usage;

	BufferType type{BufferType::Host};
};

class Buffer : public Resource<vk::Buffer> {
  public:
	using CreateInfo = BufferCreateInfo;

	static constexpr auto min_size_v = vk::DeviceSize{1};

	Buffer() = default;

	explicit Buffer(gsl::not_null<IRenderApi const*> api, CreateInfo const& create_info, vk::DeviceSize size = min_size_v);

	auto resize(vk::DeviceSize size) -> bool;

	auto write_in_place(BufferWrite data, vk::DeviceSize offset = 0) -> bool;
	auto resize_and_overwrite(BufferWrite data) -> bool;

	[[nodiscard]] auto get_buffer() const -> vk::Buffer { return m_buffer.get().resource; }
	[[nodiscard]] auto get_mapped() const -> void* { return m_mapped.get(); }
	[[nodiscard]] auto mapped_span() const -> std::span<std::byte>;

	[[nodiscard]] auto get_capacity() const -> vk::DeviceSize { return m_capacity; }
	[[nodiscard]] auto get_size() const -> vk::DeviceSize { return m_size; }
	[[nodiscard]] auto get_info() const -> CreateInfo const& { return m_info; }

	[[nodiscard]] auto descriptor_info() const -> vk::DescriptorBufferInfo;

  private:
	struct Deleter {
		void operator()(Payload const& buffer) const noexcept;
	};

	CreateInfo m_info{};
	klib::Unique<Payload, Deleter> m_buffer{};
	vk::DeviceSize m_capacity{};
	vk::DeviceSize m_size{};
	klib::Unique<void*> m_mapped{};
};

enum class ImageFlag : std::int8_t {
	None = 0,
	DedicatedAlloc = 1 << 0,
	MipMapped = 1 << 1,
};
using ImageFlags = klib::EnumFlags<ImageFlag>;

struct ImageCreateInfo {
	static constexpr auto implicit_usage_v = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	vk::Format format;

	vk::ImageAspectFlags aspect{vk::ImageAspectFlagBits::eColor};
	vk::ImageUsageFlags usage{implicit_usage_v};
	vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
	std::uint32_t layers{1};
	vk::ImageViewType view_type{vk::ImageViewType::e2D};
	ImageFlags flags{};
};

class Image : public Resource<vk::Image> {
  public:
	using CreateInfo = ImageCreateInfo;

	static constexpr auto min_extent_v = vk::Extent2D{1, 1};

	Image() = default;

	explicit Image(gsl::not_null<IRenderApi const*> api, CreateInfo const& create_info, vk::Extent2D extent = min_extent_v);

	auto resize(vk::Extent2D extent) -> bool;
	void transition(vk::CommandBuffer command_buffer, vk::ImageMemoryBarrier2 barrier);

	auto resize_and_overwrite(std::span<Bitmap const> layers) -> bool;
	auto resize_and_overwrite(Bitmap bitmap) -> bool;

	[[nodiscard]] auto get_image() const -> vk::Image { return m_image.get().resource; }
	[[nodiscard]] auto get_view() const -> vk::ImageView { return *m_view; }

	[[nodiscard]] auto get_extent() const -> vk::Extent2D { return m_extent; }
	[[nodiscard]] auto get_layout() const -> vk::ImageLayout { return m_layout; }
	[[nodiscard]] auto get_info() const -> CreateInfo const& { return m_info; }
	[[nodiscard]] auto get_mip_levels() const -> std::uint32_t { return m_mip_levels; }
	[[nodiscard]] auto subresource_range() const -> vk::ImageSubresourceRange;

	[[nodiscard]] auto render_target() const -> RenderTarget { return RenderTarget{.image = get_image(), .view = get_view(), .extent = get_extent()}; }

  private:
	struct Deleter {
		void operator()(Payload const& image) const noexcept;
	};

	CreateInfo m_info{};
	std::uint32_t m_mip_levels{1};
	klib::Unique<Payload, Deleter> m_image{};
	vk::UniqueImageView m_view{};
	vk::Extent2D m_extent{};
	vk::ImageLayout m_layout{};
};

[[nodiscard]] constexpr auto create_sampler_ci(vk::SamplerAddressMode const wrap, vk::Filter const filter) {
	auto ret = vk::SamplerCreateInfo{};
	ret.setAddressModeU(wrap)
		.setAddressModeV(wrap)
		.setAddressModeW(wrap)
		.setMinFilter(filter)
		.setMagFilter(filter)
		.setMaxLod(VK_LOD_CLAMP_NONE)
		.setBorderColor(vk::BorderColor::eFloatTransparentBlack)
		.setMipmapMode(vk::SamplerMipmapMode::eNearest);
	return ret;
}

constexpr auto sampler_ci_v = create_sampler_ci(vk::SamplerAddressMode::eClampToEdge, vk::Filter::eLinear);

struct TextureCreateInfo {
	vk::Format format{vk::Format::eR8G8B8A8Srgb};
	vk::ImageAspectFlagBits aspect{vk::ImageAspectFlagBits::eColor};
	vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
	ImageFlags flags{ImageFlag::MipMapped};
	vk::SamplerCreateInfo sampler{sampler_ci_v};
};

class Texture {
  public:
	using CreateInfo = TextureCreateInfo;

	explicit Texture(gsl::not_null<IRenderApi const*> api, Bitmap const& bitmap = {}, CreateInfo const& create_info = {});

	[[nodiscard]] auto get_extent() const -> vk::Extent2D { return m_image.get_extent(); }
	[[nodiscard]] auto get_image() const -> Image const& { return m_image; }
	[[nodiscard]] auto get_sampler() const -> vk::Sampler { return *m_sampler; }

	[[nodiscard]] auto descriptor_info() const -> vk::DescriptorImageInfo;

  private:
	Image m_image{};
	vk::UniqueSampler m_sampler{};
};
} // namespace kvf::vma
