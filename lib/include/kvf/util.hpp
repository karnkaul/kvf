#pragma once
#include "GLFW/glfw3.h"
#include "kvf/color.hpp"
#include "kvf/rect.hpp"
#include <vulkan/vulkan.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <gsl/pointers>
#include <string>

using namespace std::chrono_literals;

namespace kvf::util {
inline constexpr auto srgb_formats_v = std::array{vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb, vk::Format::eA8B8G8R8SrgbPack32};
inline constexpr auto linear_formats_v = std::array{vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8A8Unorm, vk::Format::eA8B8G8R8UnormPack32};

[[nodiscard]] constexpr auto is_linear(vk::Format const format) -> bool { return std::ranges::find(linear_formats_v, format) != linear_formats_v.end(); }
[[nodiscard]] constexpr auto is_srgb(vk::Format const format) -> bool { return std::ranges::find(srgb_formats_v, format) != srgb_formats_v.end(); }

[[nodiscard]] constexpr auto to_string_view(vk::PresentModeKHR const present_mode) -> std::string_view {
	switch (present_mode) {
	case vk::PresentModeKHR::eFifo: return "FIFO";
	case vk::PresentModeKHR::eFifoRelaxed: return "FIFO Relaxed";
	case vk::PresentModeKHR::eMailbox: return "Mailbox";
	case vk::PresentModeKHR::eImmediate: return "Immediate";
	default: return "Unsupported";
	}
}

template <typename Type = float>
constexpr auto to_glm_vec(vk::Extent2D const in) -> glm::vec<2, Type> {
	return glm::vec<2, Type>{in.width, in.height};
}

template <typename Type>
constexpr auto to_vk_extent(glm::vec<2, Type> const in) -> vk::Extent2D {
	return {std::uint32_t(in.x), std::uint32_t(in.y)};
}

constexpr auto scale_extent(vk::Extent2D const extent, float const scale) -> vk::Extent2D {
	return vk::Extent2D{std::uint32_t(float(extent.width) * scale), std::uint32_t(float(extent.height) * scale)};
}

constexpr auto ndc_to_uv(glm::vec2 const ndc) -> glm::vec2 { return {ndc.x + 0.5f, 0.5f - ndc.y}; }
constexpr auto uv_to_ndc(glm::vec2 const ndc) -> glm::vec2 { return {ndc.x - 0.5f, 0.5f - ndc.y}; }
constexpr auto ndc_to_uv(UvRect const& rect) { return UvRect{.lt = ndc_to_uv(rect.lt), .rb = ndc_to_uv(rect.rb)}; }
constexpr auto uv_to_ndc(UvRect const& rect) { return UvRect{.lt = uv_to_ndc(rect.lt), .rb = uv_to_ndc(rect.rb)}; }

template <klib::NumberT Type>
constexpr void ensure_positive(Type& out) {
	out = std::max(out, Type(1));
}
constexpr void ensure_positive(vk::Extent2D& out) {
	ensure_positive(out.width);
	ensure_positive(out.height);
}

constexpr auto is_norm(float const f) -> bool { return f >= 0.0f && f <= 1.0f; }
constexpr auto is_norm(glm::vec2 const v) -> bool { return is_norm(v.x) && is_norm(v.y); }
constexpr auto is_norm(UvRect const& r) -> bool { return is_norm(r.lt) && is_norm(r.rb); }

[[nodiscard]] constexpr auto create_sampler_ci(vk::SamplerAddressMode const wrap, vk::Filter const filter, float const aniso = 16.0f) {
	auto ret = vk::SamplerCreateInfo{};
	ret.setAddressModeU(wrap)
		.setAddressModeV(wrap)
		.setAddressModeW(wrap)
		.setMinFilter(filter)
		.setMagFilter(filter)
		.setMaxAnisotropy(aniso)
		.setMaxLod(VK_LOD_CLAMP_NONE)
		.setBorderColor(vk::BorderColor::eFloatTransparentBlack)
		.setMipmapMode(vk::SamplerMipmapMode::eNearest);
	return ret;
}

[[nodiscard]] auto window_size(gsl::not_null<GLFWwindow*> window) -> glm::ivec2;
[[nodiscard]] auto framebuffer_size(gsl::not_null<GLFWwindow*> window) -> glm::ivec2;
[[nodiscard]] auto is_window_closing(gsl::not_null<GLFWwindow*> window) -> bool;
void set_window_should_close(gsl::not_null<GLFWwindow*> window, bool value);

[[nodiscard]] auto color_from_hex(std::string_view hex) -> Color;
[[nodiscard]] auto to_hex_string(Color const& color) -> std::string;

[[nodiscard]] auto compute_mip_levels(vk::Extent2D extent) -> std::uint32_t;

[[nodiscard]] auto ubo_write(gsl::not_null<vk::DescriptorBufferInfo const*> info, vk::DescriptorSet set, std::uint32_t binding) -> vk::WriteDescriptorSet;
[[nodiscard]] auto ssbo_write(gsl::not_null<vk::DescriptorBufferInfo const*> info, vk::DescriptorSet set, std::uint32_t binding) -> vk::WriteDescriptorSet;
[[nodiscard]] auto image_write(gsl::not_null<vk::DescriptorImageInfo const*> info, vk::DescriptorSet set, std::uint32_t binding) -> vk::WriteDescriptorSet;

auto wait_for_fence(vk::Device device, vk::Fence fence, std::chrono::nanoseconds timeout = 5s) -> bool;

void record_barriers(vk::CommandBuffer command_buffer, std::span<vk::ImageMemoryBarrier2 const> image_barriers);

inline void record_barrier(vk::CommandBuffer const command_buffer, vk::ImageMemoryBarrier2 const& image_barrier) {
	record_barriers(command_buffer, {&image_barrier, 1});
}

struct ImageViewCreateInfo {
	vk::Image image;
	vk::Format format;

	vk::ImageSubresourceRange subresource{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageViewType type{vk::ImageViewType::e2D};
};
[[nodiscard]] auto create_image_view(vk::Device device, ImageViewCreateInfo const& create_info) -> vk::UniqueImageView;
} // namespace kvf::util
