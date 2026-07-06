#include "detail/vma.hpp"
#include "klib/debug/assert.hpp"
#include "kvf/panic.hpp"
#include "kvf/util.hpp"

namespace kvf::detail {
namespace vma {
void Buffer::Deleter::operator()(Buffer const& buffer) const noexcept { vmaDestroyBuffer(buffer.allocator, buffer.buffer, buffer.allocation); }

void Image::Deleter::operator()(Image const& image) const noexcept { vmaDestroyImage(image.allocator, image.image, image.allocation); }
} // namespace vma

auto vma::create_buffer(VmaAllocator allocator, BufferCreateInfo const& create_info) noexcept(false) -> UniqueBuffer {
	KLIB_ASSERT(create_info.type == BufferType::Host || (create_info.usage & vk::BufferUsageFlagBits::eTransferDst) == vk::BufferUsageFlagBits::eTransferDst);
	KLIB_ASSERT(create_info.size > 0);

	auto allocation_ci = VmaAllocationCreateInfo{};
	allocation_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	if (create_info.type == BufferType::Device) {
		allocation_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	} else {
		allocation_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocation_ci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
	}

	auto const buffer_ci = vk::BufferCreateInfo{{}, create_info.size, create_info.usage};
	auto c_buffer_ci = static_cast<VkBufferCreateInfo>(buffer_ci);

	VmaAllocation allocation{};
	VkBuffer buffer{};
	auto allocation_info = VmaAllocationInfo{};
	if (vmaCreateBuffer(allocator, &c_buffer_ci, &allocation_ci, &buffer, &allocation, &allocation_info) != VK_SUCCESS) {
		throw Panic{"Failed to create Vulkan Buffer"};
	}

	return Buffer{.buffer = buffer, .allocator = allocator, .allocation = allocation, .mapped = allocation_info.pMappedData};
}

auto vma::create_image(VmaAllocator allocator, std::uint32_t const queue_family, ImageCreateInfo const& create_info) -> UniqueImage {
	KLIB_ASSERT((create_info.usage & ImageCreateInfo::implicit_usage_v) == ImageCreateInfo::implicit_usage_v);
	KLIB_ASSERT(create_info.format != vk::Format::eUndefined);
	KLIB_ASSERT(create_info.extent.width > 0 && create_info.extent.height > 0);

	auto const mip_mapped = (create_info.flags & ImageFlag::MipMapped) == ImageFlag::MipMapped;
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
	if (vmaCreateImage(allocator, &vici, &vaci, &image, &allocation, {}) != VK_SUCCESS) { throw Panic{"Failed to create Vulkan Image"}; }

	return Image{.image = image, .allocator = allocator, .allocation = allocation, .mip_levels = image_ci.mipLevels};
}
} // namespace kvf::detail
