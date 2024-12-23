#pragma once
#include <kvf/pipeline_state.hpp>
#include <string_view>

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

void record_barriers(vk::CommandBuffer command_buffer, std::span<vk::ImageMemoryBarrier2 const> image_barriers);

inline void record_barrier(vk::CommandBuffer const command_buffer, vk::ImageMemoryBarrier2 const& image_barrier) {
	record_barriers(command_buffer, {&image_barrier, 1});
}

[[nodiscard]] auto create_shader_module(vk::Device device, std::span<std::uint32_t const> spir_v) -> vk::UniqueShaderModule;
[[nodiscard]] auto create_pipeline(vk::Device device, vk::PipelineLayout layout, PipelineState const& state) -> vk::UniquePipeline;
} // namespace kvf::util
