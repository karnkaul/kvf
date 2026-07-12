#include "detail/render_image.hpp"
#include "detail/render_buffer.hpp"
#include "klib/debug/assert.hpp"
#include "kvf/is_positive.hpp"
#include "kvf/render_buffer.hpp"
#include "kvf/render_device.hpp"
#include "kvf/scratch_command_buffer.hpp"
#include "kvf/util.hpp"
#include "log.hpp"
#include <stb/stb_image_write.h>
#include <algorithm>
#include <ranges>

namespace kvf::detail {
namespace {
class MipMapCreator {
  public:
	explicit MipMapCreator(IRenderImage& image, IRenderDevice const& render_device, vk::CommandBuffer command_buffer)
		: m_image(image), m_render_device(render_device), m_command_buffer(command_buffer) {}

	void record() {
		m_barrier = m_render_device.create_image_barrier(m_image.get_aspect());
		m_barrier.setImage(m_image.get_image())
			.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
			.setDstAccessMask(m_barrier.srcAccessMask)
			.setDstStageMask(m_barrier.srcStageMask)
			.setOldLayout(m_image.get_layout())
			.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
		m_barrier.subresourceRange.setAspectMask(m_image.get_aspect()).setLevelCount(1).setLayerCount(m_image.get_layers());
		util::record_barrier(m_command_buffer, m_barrier);

		auto src_extent = vk::Extent3D{m_image.get_extent(), 1};
		for (std::uint32_t mip = 0; mip + 1 < m_image.get_mip_levels(); ++mip) {
			vk::Extent3D dst_extent = vk::Extent3D(std::max(src_extent.width / 2, 1u), std::max(src_extent.height / 2, 1u), 1u);
			auto const src_offset = vk::Offset3D{static_cast<int>(src_extent.width), static_cast<int>(src_extent.height), 1};
			auto const dst_offset = vk::Offset3D{static_cast<int>(dst_extent.width), static_cast<int>(dst_extent.height), 1};
			blit_next_mip(mip, src_offset, dst_offset);
			src_extent = dst_extent;
		}
	}

  private:
	auto blit_mips(std::uint32_t const src_level, vk::Offset3D const src_offset, vk::Offset3D const dst_offset) const -> void {
		auto ib = vk::ImageBlit2{};
		ib.srcSubresource.setAspectMask(m_image.get_aspect()).setMipLevel(src_level).setLayerCount(m_image.get_layers());
		ib.dstSubresource.setAspectMask(m_image.get_aspect()).setMipLevel(src_level + 1).setLayerCount(m_image.get_layers());
		ib.srcOffsets[1] = src_offset;
		ib.dstOffsets[1] = dst_offset;
		auto bii = vk::BlitImageInfo2{};
		bii.setSrcImage(m_barrier.image)
			.setDstImage(m_barrier.image)
			.setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
			.setRegions(ib)
			.setFilter(vk::Filter::eLinear);
		m_command_buffer.blitImage2(bii);
	}

	auto blit_next_mip(std::uint32_t const src_level, vk::Offset3D const src_offset, vk::Offset3D const dst_offset) -> void {
		m_barrier.subresourceRange.setBaseMipLevel(src_level + 1);
		m_barrier.setOldLayout(vk::ImageLayout::eUndefined).setNewLayout(vk::ImageLayout::eTransferDstOptimal);
		util::record_barrier(m_command_buffer, m_barrier);

		blit_mips(src_level, src_offset, dst_offset);

		m_barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal).setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
		util::record_barrier(m_command_buffer, m_barrier);
	}

	IRenderImage& m_image;
	IRenderDevice const& m_render_device;
	vk::CommandBuffer m_command_buffer{};
	vk::ImageMemoryBarrier2 m_barrier{};
};

template <typename FlagsT, typename BitsT>
[[nodiscard]] constexpr auto is_set(FlagsT const flags, BitsT const bit) -> bool {
	return (flags & bit) == bit;
}

[[nodiscard]] constexpr auto is_copyable(vk::Format const format) { return format == vk::Format::eR8G8B8A8Srgb || format == vk::Format::eR8G8B8A8Unorm; }
[[nodiscard]] constexpr auto is_copyable(vk::ImageAspectFlags const aspect) { return is_set(aspect, vk::ImageAspectFlagBits::eColor); }
[[nodiscard]] constexpr auto is_copyable(vk::ImageLayout const layout) { return layout != vk::ImageLayout::eUndefined; }
} // namespace

RenderImage::RenderImage(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info) : m_render_device(render_device) {
	recreate_impl(create_info);
}

void RenderImage::resize(vk::Extent2D extent) {
	util::ensure_positive(extent);
	if (extent == m_info.extent) { return; }

	auto info = m_info;
	info.extent = extent;
	recreate(info);
}

auto RenderImage::resize_and_overwrite(std::span<Bitmap const> layers) -> bool {
	if (layers.empty()) { return false; }

	auto size = layers.front().size;
	if (!is_positive(size)) {
		layers = {&white_bitmap_v, 1};
		size = {1, 1};
	}

	auto const extent = util::to_vk_extent(size);

	auto const layer_size = vk::DeviceSize(extent.width * extent.height * Bitmap::channels_v);
	KLIB_ASSERT(layer_size > 0);

	if (!m_image.get().image || m_info.layers != std::uint32_t(layers.size())) {
		m_info.layers = std::uint32_t(layers.size());
		m_info.extent = extent;
		recreate(m_info);
	}

	auto const total_size = layers.size() * layer_size;
	auto const check = [size, layer_size](Bitmap const& b) { return b.size == size && b.bytes.size() == layer_size; };
	if (!std::ranges::all_of(layers, check)) { return false; }

	resize(extent);

	auto const original_layout = get_layout();

	auto const buffer_ci = BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eTransferSrc,
		.type = BufferType::Host,
		.size = total_size,
	};
	auto buffer = detail::RenderBuffer{m_render_device, buffer_ci};
	auto& staging = static_cast<IRenderBuffer&>(buffer);

	auto cmd = ScratchCommandBuffer{m_render_device};
	auto barrier = vk::ImageMemoryBarrier2{};
	barrier.setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setOldLayout(vk::ImageLayout::eUndefined)
		.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
	transition(cmd, barrier);

	auto span = staging.get_mapped_span();
	auto buffer_offset = vk::DeviceSize{};
	auto cbtii = vk::CopyBufferToImageInfo2{};
	cbtii.setDstImage(get_image()).setDstImageLayout(vk::ImageLayout::eTransferDstOptimal).setSrcBuffer(staging.get_buffer());
	for (auto const [index, layer] : std::views::enumerate(layers)) {
		std::memcpy(span.data(), layer.bytes.data(), layer_size);

		auto bic = vk::BufferImageCopy2{};
		bic.setImageExtent({extent.width, extent.height, 1})
			.setImageSubresource(vk::ImageSubresourceLayers{m_info.aspect, 0, std::uint32_t(index), 1})
			.setBufferOffset(buffer_offset);
		cbtii.setRegions(bic);
		cmd.get().copyBufferToImage2(cbtii);

		span = span.subspan(layer_size);
		buffer_offset += layer_size;
	}

	auto current_layout = get_layout();
	if (get_mip_levels() > 1) {
		MipMapCreator{*this, *m_render_device, cmd}.record();
		current_layout = vk::ImageLayout::eTransferSrcOptimal;
	}

	auto const final_layout = original_layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eShaderReadOnlyOptimal : original_layout;
	barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
		.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
		.setOldLayout(current_layout)
		.setNewLayout(final_layout);
	transition(cmd, barrier);

	return cmd.submit_and_wait();
}

auto RenderImage::copy_to_bitmap(vk::Extent2D custom_extent) const -> std::optional<ColorBitmap> {
	if (m_info.layers != 1 || !is_copyable(m_layout) || !is_copyable(m_info.format) || !is_copyable(m_info.aspect)) {
		log.warn("RenderImage: Invalid layers/layout/format/aspect for copying");
		return {};
	}

	auto const format_properties = m_render_device->get_gpu().device.getFormatProperties(m_info.format);
	if (!is_set(format_properties.optimalTilingFeatures, vk::FormatFeatureFlagBits::eBlitSrc) ||
		!is_set(format_properties.linearTilingFeatures, vk::FormatFeatureFlagBits::eBlitDst)) {
		log.warn("RenderImage: Device does not support blits to/from image format");
		return {};
	}

	if (custom_extent.width == 0 || custom_extent.height == 0) { custom_extent = m_info.extent; }

	auto command_buffer = ScratchCommandBuffer{m_render_device};
	auto const dst_image = blit_for_copy(command_buffer, custom_extent);
	command_buffer.submit_and_wait();

	KLIB_ASSERT(dst_image.get().mapped);

	auto const subresource_layout = m_render_device->get_device().getImageSubresourceLayout(dst_image.get().image, {m_info.aspect});
	auto const byte_count = std::size_t(subresource_layout.size + subresource_layout.offset);
	auto bytes = std::span{static_cast<std::byte const*>(dst_image.get().mapped), byte_count}.subspan(subresource_layout.offset);

	auto const image_size = util::to_glm_vec<int>(custom_extent);
	auto pixels = std::vector<Color>{};
	for (auto y = 0; y < image_size.y; y++) {
		void const* ptr = bytes.data();
		auto const row = std::span{static_cast<Color const*>(ptr), std::size_t(image_size.x)};
		pixels.append_range(row);
		bytes = bytes.subspan(subresource_layout.rowPitch);
	}

	return ColorBitmap{std::move(pixels), image_size};
}

void RenderImage::transition(vk::CommandBuffer const command_buffer, vk::ImageMemoryBarrier2 barrier) {
	barrier.setImage(get_image())
		.setSrcQueueFamilyIndex(m_render_device->get_queue_family())
		.setDstQueueFamilyIndex(barrier.srcQueueFamilyIndex)
		.setSubresourceRange(subresource_range());
	util::record_barrier(command_buffer, barrier);
	m_layout = barrier.newLayout;
}

void RenderImage::recreate_impl(CreateInfo create_info) {
	create_info.usage |= CreateInfo::implicit_usage_v;
	if (create_info.format == vk::Format::eUndefined) { create_info.format = vk::Format::eR8G8B8A8Srgb; }
	util::ensure_positive(create_info.extent);

	if (create_info.extent.width == 1 || create_info.extent.height == 1) { create_info.flags &= ~ImageFlag::MipMaps; }
	m_image = vma::create_image(m_render_device->get_allocator(), m_render_device->get_queue_family(), create_info);
	m_info = create_info;

	auto const image_view_ci = util::ImageViewCreateInfo{
		.image = m_image.get().image,
		.format = create_info.format,
		.subresource = subresource_range(),
		.type = create_info.view_type,
	};
	m_image_view = util::create_image_view(m_render_device->get_device(), image_view_ci);
	m_layout = vk::ImageLayout::eUndefined;
}

auto RenderImage::get_pre_render_barrier() -> vk::ImageMemoryBarrier2 {
	m_layout = vk::ImageLayout::eAttachmentOptimal;
	auto ret = m_render_device->create_image_barrier(m_info.aspect);
	ret.setImage(m_image.get().image)
		.setSrcAccessMask(vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eTransferRead)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eTransfer)
		.setNewLayout(m_layout)
		.setOldLayout(vk::ImageLayout::eUndefined);
	if (m_info.aspect == vk::ImageAspectFlagBits::eDepth) {
		ret.setDstAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite).setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests);
	} else {
		ret.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite).setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	}
	return ret;
}

auto RenderImage::get_post_render_barrier() -> vk::ImageMemoryBarrier2 {
	m_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	auto ret = m_render_device->create_image_barrier(m_info.aspect);
	ret.setImage(m_image.get().image)
		.setOldLayout(vk::ImageLayout::eAttachmentOptimal)
		.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eTransferRead)
		.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eTransfer)
		.setNewLayout(m_layout);
	if (m_info.aspect == vk::ImageAspectFlagBits::eDepth) {
		ret.setSrcAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite).setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader);
	} else {
		ret.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite).setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	}
	return ret;
}

auto RenderImage::blit_for_copy(vk::CommandBuffer const command_buffer, vk::Extent2D const extent) const -> vma::UniqueImage {
	KLIB_ASSERT(m_layout != vk::ImageLayout::eUndefined);

	auto const dst_image_ci = CreateInfo{
		.format = m_info.format,
		.extent = extent,
	};
	auto dst_image = vma::create_image_for_copy(m_image.get().allocator, m_render_device->get_queue_family(), dst_image_ci);

	auto barriers = std::array<vk::ImageMemoryBarrier2, 2>{};
	barriers[0] = m_render_device->create_image_barrier();
	barriers[1] = barriers[0];
	barriers[0].setImage(m_image.get().image);
	barriers[1].setImage(dst_image.get().image);

	barriers[0]
		.setOldLayout(m_layout)
		.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
		.setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
	barriers[1]
		.setOldLayout(vk::ImageLayout::eUndefined)
		.setNewLayout(vk::ImageLayout::eTransferDstOptimal)
		.setSrcAccessMask(vk::AccessFlagBits2::eNone)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
	util::record_barriers(command_buffer, barriers);

	static constexpr auto to_offset = [](vk::Extent2D const extent) { return vk::Offset3D{std::int32_t(extent.width), std::int32_t(extent.height), 1}; };

	auto subresource_layers = vk::ImageSubresourceLayers{};
	subresource_layers.setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);

	auto image_blit = vk::ImageBlit2{};
	image_blit.setSrcOffsets({vk::Offset3D{}, to_offset(m_info.extent)})
		.setDstOffsets({vk::Offset3D{}, to_offset(extent)})
		.setSrcSubresource(subresource_layers)
		.setDstSubresource(subresource_layers);

	auto blit_info = vk::BlitImageInfo2{};
	blit_info.setSrcImage(m_image.get().image)
		.setDstImage(dst_image.get().image)
		.setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
		.setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
		.setFilter(vk::Filter::eNearest)
		.setRegions(image_blit);
	command_buffer.blitImage2(blit_info);

	barriers[0]
		.setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
		.setNewLayout(m_layout)
		.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead)
		.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
	barriers[1]
		.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
		.setNewLayout(vk::ImageLayout::eGeneral)
		.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
		.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
	util::record_barriers(command_buffer, barriers);

	return dst_image;
}
} // namespace kvf::detail

namespace kvf {
auto IRenderImage::create(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info) -> std::unique_ptr<IRenderImage> {
	return std::make_unique<detail::RenderImage>(render_device, create_info);
}

auto IRenderImage::create_texture(gsl::not_null<IRenderDevice*> render_device, Bitmap bitmap, bool const mip_map) -> std::unique_ptr<IRenderImage> {
	auto image_ci = ImageCreateInfo{
		.format = vk::Format::eR8G8B8A8Srgb,
		.aspect = vk::ImageAspectFlagBits::eColor,
		.view_type = vk::ImageViewType::e2D,
		.extent = util::to_vk_extent(bitmap.size),
	};
	if (mip_map) {
		image_ci.flags |= ImageFlag::MipMaps;
	} else {
		image_ci.flags &= ~ImageFlag::MipMaps;
	}
	auto ret = create(render_device, image_ci);
	if (bitmap.bytes.empty() || !is_positive(bitmap.size)) { bitmap = white_bitmap_v; }
	ret->resize_and_overwrite(bitmap);
	return ret;
}

auto IRenderImage::subresource_range() const -> vk::ImageSubresourceRange {
	return vk::ImageSubresourceRange{get_aspect(), 0, get_mip_levels(), 0, get_layers()};
}

auto IRenderImage::descriptor_info(vk::Sampler const sampler) const -> vk::DescriptorImageInfo {
	auto ret = vk::DescriptorImageInfo{};
	ret.setImageView(get_image_view()).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal).setSampler(sampler);
	return ret;
}
} // namespace kvf
