#pragma once
#include "klib/base_types.hpp"
#include "klib/version.hpp"
#include "kvf/frame_index.hpp"
#include "kvf/gpu.hpp"
#include "kvf/pipeline_state.hpp"
#include "kvf/render_api.hpp"
#include "kvf/render_device_fwd.hpp"
#include "kvf/render_target.hpp"
#include "kvf/vma.hpp"
#include <GLFW/glfw3.h>
#include <cstdint>
#include <gsl/pointers>
#include <memory>
#include <span>

namespace kvf {
enum class RenderDeviceFlag : std::uint8_t {
	None = 0,
	LinearBackbuffer = 1 << 0,
	ShaderObjectFeature = 1 << 1,
	ShaderObjectLayer = 1 << 2,
};
constexpr auto enable_enum_bitops(RenderDeviceFlag /*unused*/) { return true; }

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

	RenderDeviceFlag flags{};
	std::span<vk::DescriptorPoolSize const> custom_pool_sizes{};
	std::uint32_t sets_per_pool{sets_per_pool_v};
	GpuSelector const* gpu_selector{nullptr};
};

struct ShaderObjectCreateInfo {
	std::span<std::uint32_t const> vertex_spir_v{};
	std::span<std::uint32_t const> fragment_spir_v{};
	std::span<vk::DescriptorSetLayout const> set_layouts{};
};

class RenderDevice : public IRenderApi, public klib::Pinned {
  public:
	static constexpr auto vk_api_version_v = klib::Version{.major = 1, .minor = 3};

	static constexpr auto present_modes_v = std::array{
		vk::PresentModeKHR::eFifo,
		vk::PresentModeKHR::eFifoRelaxed,
		vk::PresentModeKHR::eMailbox,
		vk::PresentModeKHR::eImmediate,
	};

	using CreateInfo = RenderDeviceCreateInfo;

	explicit RenderDevice(gsl::not_null<GLFWwindow*> window, CreateInfo const& create_info = {});

	[[nodiscard]] auto get_window() const -> GLFWwindow*;
	[[nodiscard]] auto get_flags() const -> RenderDeviceFlag;
	[[nodiscard]] auto get_frame_index() const -> FrameIndex;

	[[nodiscard]] auto get_loader_api_version() const -> klib::Version;
	[[nodiscard]] auto get_instance() const -> vk::Instance;
	[[nodiscard]] auto get_surface() const -> vk::SurfaceKHR;
	[[nodiscard]] auto get_gpu() const -> Gpu const& final;
	[[nodiscard]] auto get_device() const -> vk::Device final;
	[[nodiscard]] auto get_queue_family() const -> std::uint32_t final;
	[[nodiscard]] auto get_allocator() const -> VmaAllocator final;

	[[nodiscard]] auto get_framebuffer_extent() const -> vk::Extent2D;
	[[nodiscard]] auto get_present_mode() const -> vk::PresentModeKHR;
	[[nodiscard]] auto get_supported_present_modes() const -> std::span<vk::PresentModeKHR const>;
	auto set_present_mode(vk::PresentModeKHR desired) -> bool;

	[[nodiscard]] auto get_swapchain_format() const -> vk::Format final;
	[[nodiscard]] auto get_depth_format() const -> vk::Format final;
	[[nodiscard]] auto is_linear_backbuffer() const -> bool final;
	[[nodiscard]] auto image_barrier(vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) const -> vk::ImageMemoryBarrier2 final;

	[[nodiscard]] auto create_sampler(vk::SamplerCreateInfo const& create_info) const -> vk::UniqueSampler final;
	[[nodiscard]] auto create_buffer(vma::BufferCreateInfo const& create_info, vk::DeviceSize size) const -> vma::Buffer;
	[[nodiscard]] auto create_image(vma::ImageCreateInfo const& create_info, vk::Extent2D extent) const -> vma::Image;
	[[nodiscard]] auto create_texture(Bitmap const& bitmap, vma::TextureCreateInfo const& create_info = {}) const -> vma::Texture;

	[[nodiscard]] auto create_pipeline(vk::PipelineLayout layout, PipelineState const& state, PipelineFormat format) const -> vk::UniquePipeline;
	[[nodiscard]] auto create_shader_objects(ShaderObjectCreateInfo const& create_info) const -> std::array<vk::UniqueShaderEXT, 2>;
	auto allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> bool;

	void queue_submit(vk::SubmitInfo2 const& si, vk::Fence fence = {}) const final;

	[[nodiscard]] auto get_render_imgui() const -> bool;
	void set_render_imgui(bool should_render);

	[[nodiscard]] auto is_window_closing() const -> bool;
	void set_window_closing(bool value) const;

	auto next_frame() -> vk::CommandBuffer;
	auto render(RenderTarget const& frame, vk::Filter filter = vk::Filter::eLinear) -> bool;

  private:
	struct Impl;
	struct Deleter {
		void operator()(Impl* ptr) const noexcept;
	};

	std::unique_ptr<Impl, Deleter> m_impl{};
};
} // namespace kvf

#include "klib/base_types.hpp"
#include "klib/enum/bitops.hpp"
#include "klib/ptr.hpp"
#include "klib/version.hpp"
#include "kvf/frame_index.hpp"
#include "kvf/gpu.hpp"
#include "kvf/image.hpp"
#include "kvf/next_frame_listener.hpp"
#include "kvf/render_pass.hpp"
#include "kvf/ring_buffer_allocator.hpp"
#include "kvf/ring_descriptor_allocator.hpp"
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <array>
#include <cstdint>
#include <gsl/pointers>
#include <memory>
#include <span>

namespace kvf::two {
enum class RenderDeviceFlag : std::uint8_t {
	None = 0,
	LinearBackbuffer = 1 << 0,
	ShaderObjectFeature = 1 << 1,
	ShaderObjectLayer = 1 << 2,
};
[[maybe_unused]] constexpr auto enable_enum_bitops(RenderDeviceFlag /*unused*/) { return true; }

inline constexpr auto supported_present_modes_v = std::array{
	vk::PresentModeKHR::eFifo,
	vk::PresentModeKHR::eFifoRelaxed,
	vk::PresentModeKHR::eMailbox,
	vk::PresentModeKHR::eImmediate,
};

struct RenderDeviceCreateInfo {
	static constexpr auto sets_per_pool_v{64};

	RenderDeviceFlag flags{};
	std::span<vk::DescriptorPoolSize const> custom_pool_sizes{};
	std::uint32_t sets_per_pool{sets_per_pool_v};
	klib::Ptr<Gpu::Selector const> gpu_selector{nullptr};
};

class IRenderDevice : public klib::Polymorphic {
  public:
	using CreateInfo = RenderDeviceCreateInfo;

	[[nodiscard]] static auto create(gsl::not_null<GLFWwindow*> window, CreateInfo const& create_info) -> std::unique_ptr<IRenderDevice>;

	static constexpr auto vk_api_version_v = klib::Version{.major = 1, .minor = 3};

	[[nodiscard]] virtual auto get_window() const -> gsl::not_null<GLFWwindow*> = 0;
	[[nodiscard]] virtual auto get_instance() const -> vk::Instance = 0;
	[[nodiscard]] virtual auto get_gpu() const -> Gpu const& = 0;
	[[nodiscard]] virtual auto get_surface() const -> vk::SurfaceKHR = 0;
	[[nodiscard]] virtual auto get_device() const -> vk::Device = 0;
	[[nodiscard]] virtual auto get_queue_family() const -> std::uint32_t = 0;
	[[nodiscard]] virtual auto get_allocator() const -> VmaAllocator = 0;

	[[nodiscard]] virtual auto get_swapchain_image_extent() const -> vk::Extent2D = 0;
	[[nodiscard]] virtual auto get_swapchain_color_format() const -> vk::Format = 0;
	[[nodiscard]] virtual auto get_optimal_depth_format() const -> vk::Format = 0;

	[[nodiscard]] virtual auto get_loader_api_version() const -> klib::Version = 0;
	[[nodiscard]] virtual auto get_flags() const -> RenderDeviceFlag = 0;

	[[nodiscard]] virtual auto get_render_imgui() const -> bool = 0;
	virtual void set_render_imgui(bool should_render) = 0;

	[[nodiscard]] virtual auto get_frame_index() const -> FrameIndex = 0;
	virtual void attach_next_frame_listener(std::weak_ptr<INextFrameListener> listener) = 0;

	[[nodiscard]] virtual auto create_buffer(BufferCreateInfo const& create_info) -> std::unique_ptr<IBuffer> = 0;
	[[nodiscard]] virtual auto create_buffer_allocator(BufferUsageLayout const& usage_layout) -> std::shared_ptr<IRingBufferAllocator> = 0;
	[[nodiscard]] virtual auto get_descriptor_allocator() -> IRingDescriptorAllocator& = 0;

	[[nodiscard]] virtual auto create_image(ImageCreateInfo const& create_info) -> std::unique_ptr<IImage> = 0;

	[[nodiscard]] virtual auto create_render_pass(vk::SampleCountFlagBits samples = IRenderPass::samples_v) -> std::unique_ptr<IRenderPass> = 0;

	virtual void queue_submit(vk::SubmitInfo2 const& si, vk::Fence fence = {}) = 0;

	virtual auto next_frame() -> vk::CommandBuffer = 0;
	virtual auto render(RenderTarget const& render_target, vk::Filter filter = vk::Filter::eLinear) -> bool = 0;

	[[nodiscard]] auto create_sampler(vk::SamplerCreateInfo create_info) const -> vk::UniqueSampler;
	[[nodiscard]] auto create_texture(Bitmap const& bitmap, bool mip_map = true) -> std::unique_ptr<IImage>;
	[[nodiscard]] auto create_pipeline(vk::PipelineLayout layout, PipelineState const& state, PipelineFormat format) const -> vk::UniquePipeline;
	[[nodiscard]] auto create_shader_objects(ShaderObjectCreateInfo const& create_info) const -> std::array<vk::UniqueShaderEXT, 2>;
	[[nodiscard]] auto create_image_barrier(vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) const -> vk::ImageMemoryBarrier2KHR;
};
} // namespace kvf::two
