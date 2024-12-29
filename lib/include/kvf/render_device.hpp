#pragma once
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>
#include <klib/constants.hpp>
#include <klib/polymorphic.hpp>
#include <klib/version.hpp>
#include <kvf/buffering.hpp>
#include <kvf/descriptor_allocator.hpp>
#include <kvf/render_device_fwd.hpp>
#include <kvf/render_target.hpp>
#include <gsl/pointers>
#include <memory>
#include <span>

namespace kvf {
struct Gpu {
	vk::PhysicalDevice device{};
	vk::PhysicalDeviceProperties properties{};
	vk::PhysicalDeviceFeatures features{};
};

struct RenderDeviceFlag {
	enum : int {
		None = 0,
		ValidationLayers = 1 << 0,
		LinearBackbuffer = 1 << 1,
	};
};
using RenderDeviceFlags = std::underlying_type_t<decltype(RenderDeviceFlag::None)>;

class GpuSelector : public klib::Polymorphic {
  public:
	[[nodiscard]] virtual auto select(std::span<Gpu const> gpus) const -> Gpu const& {
		for (auto const& gpu : gpus) {
			if (gpu.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) { return gpu; }
		}
		return gpus.front();
	}
};

struct RenderDeviceCreateInfo {
	static constexpr auto sets_per_pool_v{64};

	static constexpr auto default_flags() -> RenderDeviceFlags {
		auto ret = RenderDeviceFlags{};
		if constexpr (klib::debug_v) { ret |= RenderDeviceFlag::ValidationLayers; }
		return ret;
	}

	RenderDeviceFlags flags{default_flags()};
	std::vector<vk::DescriptorPoolSize> custom_pool_sizes{};
	std::uint32_t sets_per_pool{sets_per_pool_v};
	GpuSelector const* gpu_selector{nullptr};
};

class RenderDevice {
  public:
	static constexpr auto vk_api_version_v = klib::Version{.major = 1, .minor = 3};
	static constexpr auto aniso_v = 8.0f;

	static constexpr auto present_modes_v = std::array{
		vk::PresentModeKHR::eFifo,
		vk::PresentModeKHR::eFifoRelaxed,
		vk::PresentModeKHR::eMailbox,
		vk::PresentModeKHR::eImmediate,
	};

	using CreateInfo = RenderDeviceCreateInfo;

	explicit RenderDevice(gsl::not_null<GLFWwindow*> window, CreateInfo create_info = {});

	[[nodiscard]] auto get_window() const -> GLFWwindow*;
	[[nodiscard]] auto get_flags() const -> RenderDeviceFlags;
	[[nodiscard]] auto get_frame_index() const -> FrameIndex;

	[[nodiscard]] auto get_loader_api_version() const -> klib::Version;
	[[nodiscard]] auto get_instance() const -> vk::Instance;
	[[nodiscard]] auto get_surface() const -> vk::SurfaceKHR;
	[[nodiscard]] auto get_gpu() const -> Gpu const&;
	[[nodiscard]] auto get_device() const -> vk::Device;
	[[nodiscard]] auto get_queue_family() const -> std::uint32_t;
	[[nodiscard]] auto get_allocator() const -> VmaAllocator;
	[[nodiscard]] auto get_descriptor_allocator() const -> IDescriptorAllocator&;

	[[nodiscard]] auto get_framebuffer_extent() const -> vk::Extent2D;
	[[nodiscard]] auto get_present_mode() const -> vk::PresentModeKHR;
	[[nodiscard]] auto get_supported_present_modes() const -> std::span<vk::PresentModeKHR const>;
	auto set_present_mode(vk::PresentModeKHR desired) -> bool;

	[[nodiscard]] auto get_swapchain_format() const -> vk::Format;
	[[nodiscard]] auto get_depth_format() const -> vk::Format;
	[[nodiscard]] auto image_barrier(vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) const -> vk::ImageMemoryBarrier2;
	[[nodiscard]] auto sampler_info(vk::SamplerAddressMode wrap, vk::Filter filter, float aniso = aniso_v) const -> vk::SamplerCreateInfo;

	void queue_submit(vk::SubmitInfo2 const& si, vk::Fence fence = {});

	[[nodiscard]] auto get_render_imgui() const -> bool;
	void set_render_imgui(bool should_render);

	[[nodiscard]] auto is_window_closing() const -> bool;
	void set_window_closing(bool value) const;

	[[nodiscard]] auto next_frame() -> vk::CommandBuffer;
	void render(RenderTarget const& frame, vk::Filter filter = vk::Filter::eLinear);

  private:
	struct Impl;
	struct Deleter {
		void operator()(Impl* ptr) const noexcept;
	};

	std::unique_ptr<Impl, Deleter> m_impl{};
};
} // namespace kvf
