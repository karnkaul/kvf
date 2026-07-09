#pragma once
#include "klib/unique.hpp"
#include "kvf/render_buffer.hpp"
#include "kvf/render_image.hpp"
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace kvf::detail::vma {
struct Buffer {
	struct Deleter;

	auto operator==(Buffer const&) const -> bool = default;

	vk::Buffer buffer{};
	VmaAllocator allocator{};
	VmaAllocation allocation{};
	void* mapped{};
};

struct Buffer::Deleter {
	void operator()(Buffer const& buffer) const noexcept;
};

using UniqueBuffer = klib::Unique<Buffer, Buffer::Deleter>;

[[nodiscard]] auto create_buffer(VmaAllocator allocator, BufferCreateInfo const& create_info) noexcept(false) -> UniqueBuffer;

struct Image {
	struct Deleter;

	auto operator==(Image const&) const -> bool = default;

	vk::Image image{};
	VmaAllocator allocator{};
	VmaAllocation allocation{};
	std::uint32_t mip_levels{};
	void* mapped{};
};

struct Image::Deleter {
	void operator()(Image const& image) const noexcept;
};

using UniqueImage = klib::Unique<Image, Image::Deleter>;

[[nodiscard]] auto create_image(VmaAllocator allocator, std::uint32_t queue_family, ImageCreateInfo const& create_info) noexcept(false) -> UniqueImage;
[[nodiscard]] auto create_image_for_copy(VmaAllocator allocator, std::uint32_t queue_family, ImageCreateInfo const& create_info) noexcept(false) -> UniqueImage;
} // namespace kvf::detail::vma
