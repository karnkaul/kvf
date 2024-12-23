#pragma once
#include <klib/c_string.hpp>
#include <vulkan/vulkan.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kvf {
enum class IoResult : int { Success, OpenFailed, SizeMismatch };

namespace util {
constexpr auto to_str(vk::PresentModeKHR const present_mode) -> std::string_view {
	switch (present_mode) {
	case vk::PresentModeKHR::eFifo: return "FIFO";
	case vk::PresentModeKHR::eFifoRelaxed: return "FIFO Relaxed";
	case vk::PresentModeKHR::eMailbox: return "Mailbox";
	case vk::PresentModeKHR::eImmediate: return "Immediate";
	default: return "Unsupported";
	}
}

constexpr auto scale_extent(vk::Extent2D const extent, float const scale) -> vk::Extent2D {
	return vk::Extent2D{std::uint32_t(float(extent.width) * scale), std::uint32_t(float(extent.height) * scale)};
}

void record_barriers(vk::CommandBuffer command_buffer, std::span<vk::ImageMemoryBarrier2 const> image_barriers);

inline void record_barrier(vk::CommandBuffer const command_buffer, vk::ImageMemoryBarrier2 const& image_barrier) {
	record_barriers(command_buffer, {&image_barrier, 1});
}

auto string_from_file(std::string& out_string, klib::CString path) -> IoResult;
auto bytes_from_file(std::vector<std::byte>& out_bytes, klib::CString path) -> IoResult;
auto spirv_from_file(std::vector<std::uint32_t>& out_code, klib::CString path) -> IoResult;
} // namespace util
} // namespace kvf
