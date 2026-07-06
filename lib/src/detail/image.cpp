#include "detail/image.hpp"
#include "kvf/buffer.hpp"
#include "kvf/panic.hpp"
#include "kvf/scratch_command_buffer.hpp"
#include "kvf/util.hpp"
#include <algorithm>
#include <ranges>

namespace kvf::detail {
namespace {
class MipMapCreator {
  public:
	explicit MipMapCreator(IImage& image, IRenderDevice const& render_device, vk::CommandBuffer command_buffer)
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

	IImage& m_image;
	IRenderDevice const& m_render_device;
	vk::CommandBuffer m_command_buffer{};
	vk::ImageMemoryBarrier2 m_barrier{};
};
} // namespace

Image::Image(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info) : m_render_device(render_device) { recreate_impl(create_info); }

void Image::resize(vk::Extent2D extent) {
	util::ensure_positive(extent);
	if (extent == m_info.extent) { return; }
	m_info.extent = extent;
	recreate(m_info);
}

auto Image::resize_and_overwrite(std::span<Bitmap const> layers) -> bool {
	if (layers.empty()) { return false; }

	if (!m_image || m_info.layers != std::uint32_t(layers.size())) {
		m_info.layers = std::uint32_t(layers.size());
		m_info.extent = util::to_vk_extent(layers.front().size);
		recreate(m_info);
	}

	auto const size = layers.front().size;
	auto const layer_size = vk::DeviceSize(size.x * size.y * std::int32_t(Bitmap::channels_v));
	auto const total_size = layers.size() * layer_size;
	auto const check = [size, layer_size](Bitmap const& b) { return b.size == size && b.bytes.size() == layer_size; };
	if (!std::ranges::all_of(layers, check)) { return false; }

	auto const extent = util::to_vk_extent(size);
	resize(extent);
	if (layer_size == 0) { return true; }

	auto const original_layout = get_layout();

	auto const buffer_ci = BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eTransferSrc,
		.type = BufferType::Host,
		.size = total_size,
	};
	auto staging = m_render_device->create_buffer(buffer_ci);

	auto cmd = ScratchCommandBuffer{m_render_device};
	auto barrier = vk::ImageMemoryBarrier2{};
	barrier.setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setOldLayout(vk::ImageLayout::eUndefined)
		.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
	transition(cmd, barrier);

	auto span = staging->get_mapped_span();
	auto buffer_offset = vk::DeviceSize{};
	auto cbtii = vk::CopyBufferToImageInfo2{};
	cbtii.setDstImage(get_image()).setDstImageLayout(vk::ImageLayout::eTransferDstOptimal).setSrcBuffer(staging->get_buffer());
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

void Image::transition(vk::CommandBuffer const command_buffer, vk::ImageMemoryBarrier2 barrier) {
	barrier.setImage(get_image())
		.setSrcQueueFamilyIndex(m_render_device->get_queue_family())
		.setDstQueueFamilyIndex(barrier.srcQueueFamilyIndex)
		.setSubresourceRange(subresource_range());
	util::record_barrier(command_buffer, barrier);
	m_layout = barrier.newLayout;
}

void Image::recreate_impl(CreateInfo create_info) {
	create_info.usage |= CreateInfo::implicit_usage_v;
	if (create_info.format == vk::Format::eUndefined) { create_info.format = vk::Format::eR8G8B8A8Srgb; }
	util::ensure_positive(create_info.extent);

	auto const mip_mapped = (create_info.flags & ImageFlag::MipMapped) == ImageFlag::MipMapped;
	auto const queue_family = m_render_device->get_queue_family();
	auto image_ci = vk::ImageCreateInfo{};
	image_ci.setExtent({create_info.extent.width, create_info.extent.height, 1})
		.setFormat(create_info.format)
		.setUsage(create_info.usage)
		.setImageType(vk::ImageType::e2D)
		.setArrayLayers(create_info.layers)
		.setMipLevels(mip_mapped ? util::compute_mip_levels(create_info.extent) : 1)
		.setSamples(create_info.samples)
		.setTiling(vk::ImageTiling::eOptimal)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setQueueFamilyIndices(queue_family);
	auto const vici = static_cast<VkImageCreateInfo>(image_ci);
	auto vaci = VmaAllocationCreateInfo{};
	vaci.usage = VMA_MEMORY_USAGE_AUTO;
	if ((create_info.flags & ImageFlag::DedicatedAlloc) == ImageFlag::DedicatedAlloc) {
		vaci.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		vaci.priority = 1.0f;
	}
	VkImage image{};
	VmaAllocation allocation{};
	if (vmaCreateImage(m_render_device->get_allocator(), &vici, &vaci, &image, &allocation, {}) != VK_SUCCESS) { throw Panic{"Failed to create Vulkan Image"}; }

	destroy();

	m_info = create_info;
	m_image = image;
	m_allocation = allocation;
	m_mip_levels = image_ci.mipLevels;

	auto const image_view_ci = util::ImageViewCreateInfo{
		.image = m_image,
		.format = image_ci.format,
		.subresource = subresource_range(),
		.type = vk::ImageViewType::e2D,
	};
	m_image_view = util::create_image_view(m_render_device->get_device(), image_view_ci);
	m_layout = vk::ImageLayout::eUndefined;
}

void Image::destroy() {
	m_image_view = {};

	vmaDestroyImage(m_render_device->get_allocator(), m_image, m_allocation);
	m_image = vk::Image{};
	m_allocation = {};
	m_layout = vk::ImageLayout::eUndefined;
	m_mip_levels = 0;

	m_info = {};
}
} // namespace kvf::detail

namespace kvf {
auto IImage::subresource_range() const -> vk::ImageSubresourceRange { return vk::ImageSubresourceRange{get_aspect(), 0, get_mip_levels(), 0, get_layers()}; }

auto IImage::descriptor_info(vk::Sampler const sampler) const -> vk::DescriptorImageInfo {
	auto ret = vk::DescriptorImageInfo{};
	ret.setImageView(get_image_view()).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal).setSampler(sampler);
	return ret;
}
} // namespace kvf
