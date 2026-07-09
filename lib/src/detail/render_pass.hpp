#pragma once
#include "klib/ptr.hpp"
#include "kvf/frame_index.hpp"
#include "kvf/render_image.hpp"
#include "kvf/render_pass.hpp"
#include "kvf/ring.hpp"
#include <memory>

namespace kvf::detail {
class RenderPass : public IRenderPass {
  public:
	static constexpr auto samples_v = vk::SampleCountFlagBits::e1;

	explicit RenderPass(gsl::not_null<IRenderDevice*> render_device, vk::SampleCountFlagBits samples = samples_v);

  private:
	[[nodiscard]] auto get_render_device() const -> IRenderDevice& final { return *m_render_device; }

	void set_color_target(vk::Format format = vk::Format::eUndefined) final;
	void set_depth_target() final;

	void recreate(vk::SampleCountFlagBits samples = samples_v) final;

	[[nodiscard]] auto create_graphics_pipeline(vk::PipelineLayout layout, PipelineState const& state) -> vk::UniquePipeline final;

	[[nodiscard]] auto has_color_target() const -> bool final { return m_framebuffers.front().color != nullptr; }
	[[nodiscard]] auto has_resolve_target() const -> bool final { return m_framebuffers.front().resolve != nullptr; }
	[[nodiscard]] auto has_depth_target() const -> bool final { return m_framebuffers.front().depth != nullptr; }

	[[nodiscard]] auto get_color_format() const -> vk::Format final;
	[[nodiscard]] auto get_depth_format() const -> vk::Format final;
	[[nodiscard]] auto get_samples() const -> vk::SampleCountFlagBits final { return m_samples; }

	[[nodiscard]] auto get_extent() const -> vk::Extent2D final { return m_extent; }
	[[nodiscard]] auto render_target() const -> RenderTarget const& final;

	void begin_render(vk::CommandBuffer command_buffer, vk::Extent2D extent) final;
	[[nodiscard]] auto get_command_buffer() const -> vk::CommandBuffer final { return m_command_buffer; }
	auto allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> sets_layouts) -> bool final;
	void end_render() final;

	void bind_graphics_pipeline(vk::Pipeline pipeline) const final;
	void bind_graphics_shader(IGraphicsShader const& shader) const final;

	[[nodiscard]] auto render_texture_descriptor_info(vk::Sampler sampler) const -> vk::DescriptorImageInfo final;
	[[nodiscard]] auto copy_render_texture(vk::Extent2D custom_extent) const -> ColorBitmap final;

	struct Framebuffer {
		[[nodiscard]] auto render_image() const -> klib::Ptr<IRenderImage const>;

		std::unique_ptr<IRenderImage> color{};
		std::unique_ptr<IRenderImage> resolve{};
		std::unique_ptr<IRenderImage> depth{};
	};

	struct Targets {
		RenderTarget color{};
		RenderTarget resolve{};
		RenderTarget depth{};
	};

	void set_render_targets();

	gsl::not_null<IRenderDevice*> m_render_device;
	vk::SampleCountFlagBits m_samples{};

	Ring<Framebuffer> m_framebuffers{};

	vk::CommandBuffer m_command_buffer{};
	vk::Extent2D m_extent{ImageCreateInfo::min_extent_v};

	Targets m_targets{};
	std::optional<FrameIndex> m_rendered_index{};
	std::vector<vk::ImageMemoryBarrier2> m_barriers{};
};
} // namespace kvf::detail
