#pragma once
#include <klib/c_string.hpp>
#include <kvf/bitmap.hpp>
#include <kvf/buffer_write.hpp>
#include <kvf/color.hpp>
#include <kvf/rect.hpp>
#include <vulkan/vulkan.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace kvf::util {
constexpr auto to_str(vk::PresentModeKHR const present_mode) -> std::string_view {
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

[[nodiscard]] auto color_from_hex(std::string_view hex) -> Color;
[[nodiscard]] auto to_hex_string(Color const& color) -> std::string;

auto compute_mip_levels(vk::Extent2D extent) -> std::uint32_t;

auto wait_for_fence(vk::Device device, vk::Fence fence, std::chrono::nanoseconds timeout = 5s) -> bool;

void record_barriers(vk::CommandBuffer command_buffer, std::span<vk::ImageMemoryBarrier2 const> image_barriers);

inline void record_barrier(vk::CommandBuffer const command_buffer, vk::ImageMemoryBarrier2 const& image_barrier) {
	record_barriers(command_buffer, {&image_barrier, 1});
}

auto string_from_file(std::string& out_string, klib::CString path) -> bool;
auto bytes_from_file(std::vector<std::byte>& out_bytes, klib::CString path) -> bool;
auto spirv_from_file(std::vector<std::uint32_t>& out_code, klib::CString path) -> bool;
} // namespace kvf::util
