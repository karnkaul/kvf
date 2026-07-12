#include "kvf/util.hpp"
#include "klib/file_io.hpp"
#include <algorithm>
#include <cmath>

namespace kvf {
namespace {
template <typename DescriptorInfoT>
[[nodiscard]] auto descriptor_write(vk::DescriptorType const type, DescriptorInfoT const* info, vk::DescriptorSet set, std::uint32_t const binding) {
	auto ret = vk::WriteDescriptorSet{};
	ret.setDescriptorCount(1).setDescriptorType(type).setDstSet(set).setDstBinding(binding);
	if constexpr (std::same_as<DescriptorInfoT, vk::DescriptorBufferInfo>) {
		ret.setBufferInfo(*info);
	} else {
		ret.setImageInfo(*info);
	}
	return ret;
}
} // namespace

auto util::window_size(gsl::not_null<GLFWwindow*> window) -> glm::ivec2 {
	auto ret = glm::ivec2{};
	glfwGetWindowSize(window, &ret.x, &ret.y);
	return ret;
}

auto util::framebuffer_size(gsl::not_null<GLFWwindow*> window) -> glm::ivec2 {
	auto ret = glm::ivec2{};
	glfwGetFramebufferSize(window, &ret.x, &ret.y);
	return ret;
}

auto util::is_window_closing(gsl::not_null<GLFWwindow*> window) -> bool { return glfwWindowShouldClose(window) == GLFW_TRUE; }

void util::set_window_should_close(gsl::not_null<GLFWwindow*> window, bool const value) { glfwSetWindowShouldClose(window, value ? GLFW_TRUE : GLFW_FALSE); }

auto util::color_from_hex(std::string_view hex) -> Color {
	if (hex.size() != 9 || !hex.starts_with('#')) { return {}; }
	hex = hex.substr(1);
	auto const next = [&](std::uint8_t& out) {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		auto const [ptr, ec] = std::from_chars(hex.data(), hex.data() + 2, out, 16);
		hex = hex.substr(2);
		return ec == std::errc{} && ptr == hex.data();
	};
	auto ret = Color{};
	if (!next(ret.x) || !next(ret.y) || !next(ret.z) || !next(ret.w)) { return {}; }
	return ret;
}

auto util::to_hex_string(Color const& color) -> std::string { return std::format("#{:02x}{:02x}{:02x}{:02x}", color.x, color.y, color.z, color.w); }

auto util::compute_mip_levels(vk::Extent2D const extent) -> std::uint32_t {
	return static_cast<std::uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1u;
}

auto util::ubo_write(gsl::not_null<vk::DescriptorBufferInfo const*> info, vk::DescriptorSet const set, std::uint32_t const binding) -> vk::WriteDescriptorSet {
	return descriptor_write(vk::DescriptorType::eUniformBuffer, info.get(), set, binding);
}

auto util::ssbo_write(gsl::not_null<vk::DescriptorBufferInfo const*> info, vk::DescriptorSet const set, std::uint32_t const binding) -> vk::WriteDescriptorSet {
	return descriptor_write(vk::DescriptorType::eStorageBuffer, info.get(), set, binding);
}

auto util::image_write(gsl::not_null<vk::DescriptorImageInfo const*> info, vk::DescriptorSet const set, std::uint32_t const binding) -> vk::WriteDescriptorSet {
	return descriptor_write(vk::DescriptorType::eCombinedImageSampler, info.get(), set, binding);
}

auto util::wait_for_fence(vk::Device device, vk::Fence fence, std::chrono::nanoseconds const timeout) -> bool {
	return device.waitForFences(fence, vk::True, std::uint64_t(timeout.count())) == vk::Result::eSuccess;
}

void util::record_barriers(vk::CommandBuffer const command_buffer, std::span<vk::ImageMemoryBarrier2 const> image_barriers) {
	auto di = vk::DependencyInfo{};
	di.pImageMemoryBarriers = image_barriers.data();
	di.imageMemoryBarrierCount = static_cast<std::uint32_t>(image_barriers.size());
	command_buffer.pipelineBarrier2(di);
}

auto util::create_image_view(vk::Device const device, ImageViewCreateInfo const& create_info) -> vk::UniqueImageView {
	auto image_view_ci = vk::ImageViewCreateInfo{};
	image_view_ci.setImage(create_info.image).setFormat(create_info.format).setViewType(create_info.type).setSubresourceRange(create_info.subresource);
	return device.createImageViewUnique(image_view_ci);
}
} // namespace kvf
