#pragma once
#include <vk_mem_alloc.h>
#include <klib/base_types.hpp>
#include <kvf/gpu.hpp>

namespace kvf {
class IRenderApi : public klib::Polymorphic {
  public:
	static constexpr auto aniso_v = 8.0f;

	[[nodiscard]] virtual auto get_gpu() const -> Gpu const& = 0;
	[[nodiscard]] virtual auto get_device() const -> vk::Device = 0;
	[[nodiscard]] virtual auto get_queue_family() const -> std::uint32_t = 0;
	[[nodiscard]] virtual auto get_allocator() const -> VmaAllocator = 0;

	[[nodiscard]] virtual auto get_swapchain_format() const -> vk::Format = 0;
	[[nodiscard]] virtual auto get_depth_format() const -> vk::Format = 0;

	[[nodiscard]] virtual auto image_barrier(vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) const -> vk::ImageMemoryBarrier2 = 0;

	[[nodiscard]] virtual auto create_sampler(vk::SamplerCreateInfo const& create_info) const -> vk::UniqueSampler = 0;

	virtual void queue_submit(vk::SubmitInfo2 const& submit_info, vk::Fence signal) const = 0;
};
} // namespace kvf
