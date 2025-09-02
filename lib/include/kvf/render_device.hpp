#pragma once
#include <GLFW/glfw3.h>
#include <klib/base_types.hpp>
#include <klib/constants.hpp>
#include <klib/version.hpp>
#include <kvf/buffered.hpp>
#include <kvf/gpu.hpp>
#include <kvf/pipeline_state.hpp>
#include <kvf/render_api.hpp>
#include <kvf/render_device_fwd.hpp>
#include <kvf/render_target.hpp>
#include <kvf/vma.hpp>
#include <cstdint>
#include <gsl/pointers>
#include <memory>
#include <span>

namespace kvf {
enum class RenderDeviceFlag : std::uint8_t;
} // namespace kvf

template <>
inline constexpr auto klib::enable_enum_ops_v<kvf::RenderDeviceFlag> = true;

namespace kvf {
enum class RenderDeviceFlag : std::uint8_t {
	None = 0,
	LinearBackbuffer = 1 << 0,
	ShaderObjectFeature = 1 << 1,
	ShaderObjectLayer = 1 << 2,
};

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
	[[nodiscard]] auto image_barrier(vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) const -> vk::ImageMemoryBarrier2 final;

	[[nodiscard]] auto create_sampler(vk::SamplerCreateInfo const& create_info) const -> vk::UniqueSampler final;
	[[nodiscard]] auto create_buffer(vma::BufferCreateInfo const& create_info, vk::DeviceSize size) const -> vma::Buffer;
	[[nodiscard]] auto create_image(vma::ImageCreateInfo const& create_info, vk::Extent2D extent) const -> vma::Image;
	[[nodiscard]] auto create_texture(Bitmap const& bitmap, vma::TextureCreateInfo const& create_info = {}) const -> vma::Texture;

	[[nodiscard]] auto create_pipeline(vk::PipelineLayout layout, PipelineState const& state, PipelineFormat format) const -> vk::UniquePipeline;
	[[nodiscard]] auto create_shader_objects(ShaderObjectCreateInfo const& create_info) const -> std::array<vk::UniqueShaderEXT, 2>;
	auto allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> bool;
	[[nodiscard]] auto allocate_scratch_buffer(vk::BufferUsageFlags usage, vk::DeviceSize size) -> vma::Buffer&;
	[[nodiscard]] auto scratch_descriptor_buffer(vk::BufferUsageFlags usage, BufferWrite write) -> vk::DescriptorBufferInfo;

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
