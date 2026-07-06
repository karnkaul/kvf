#pragma once
#include "kvf/buffered.hpp"
#include "kvf/pipeline_state.hpp"
#include "kvf/rect.hpp"
#include "kvf/render_device_fwd.hpp"
#include "kvf/render_pass_fwd.hpp"
#include "kvf/vma.hpp"
#include <glm/vec4.hpp>

namespace kvf {
class RenderPass {
  public:
	static constexpr auto samples_v = vk::SampleCountFlagBits::e1;

	explicit RenderPass(gsl::not_null<RenderDevice*> render_device, vk::SampleCountFlagBits samples = samples_v);

	[[nodiscard]] auto get_render_device() const -> RenderDevice& { return *m_device; }

	auto set_color_target(vk::Format format = vk::Format::eUndefined) -> RenderPass&; // undefined = RGBA with swapchain color space
	auto set_depth_target() -> RenderPass&;

	void recreate(vk::SampleCountFlagBits samples = samples_v);

	[[nodiscard]] auto create_pipeline(vk::PipelineLayout layout, PipelineState const& state) -> vk::UniquePipeline;

	[[nodiscard]] auto has_color_target() const -> bool { return bool(m_framebuffers.front().color); }
	[[nodiscard]] auto has_resolve_target() const -> bool { return bool(m_framebuffers.front().resolve); }
	[[nodiscard]] auto has_depth_target() const -> bool { return bool(m_framebuffers.front().depth); }

	[[nodiscard]] auto get_color_format() const -> vk::Format;
	[[nodiscard]] auto get_depth_format() const -> vk::Format;
	[[nodiscard]] auto get_samples() const -> vk::SampleCountFlagBits { return m_samples; }

	[[nodiscard]] auto get_extent() const -> vk::Extent2D { return m_extent; }
	[[nodiscard]] auto render_target() const -> RenderTarget const&;

	void begin_render(vk::CommandBuffer command_buffer, vk::Extent2D extent);
	[[nodiscard]] auto get_command_buffer() const -> vk::CommandBuffer { return m_command_buffer; }
	void end_render();

	[[nodiscard]] auto to_viewport(UvRect n_rect) const -> vk::Viewport;
	[[nodiscard]] auto to_scissor(UvRect n_rect) const -> vk::Rect2D;

	void bind_pipeline(vk::Pipeline pipeline) const;

	[[nodiscard]] auto render_texture_descriptor_info(vk::Sampler sampler) const -> vk::DescriptorImageInfo;

	glm::vec4 clear_color{0.0f};
	vk::ClearDepthStencilValue clear_depth{1.0f, 0};
	vk::AttachmentStoreOp depth_store_op{vk::AttachmentStoreOp::eDontCare};

  private:
	struct Framebuffer {
		vma::Image color{};
		vma::Image resolve{};
		vma::Image depth{};
	};

	struct Targets {
		RenderTarget color{};
		RenderTarget resolve{};
		RenderTarget depth{};
	};

	void set_render_targets();

	gsl::not_null<RenderDevice*> m_device;
	vk::SampleCountFlagBits m_samples{};

	Buffered<Framebuffer> m_framebuffers{};

	vk::CommandBuffer m_command_buffer{};
	vk::Extent2D m_extent{};

	Targets m_targets{};
	std::vector<vk::ImageMemoryBarrier2> m_barriers{};
};
} // namespace kvf

#include "klib/base_types.hpp"
#include "kvf/pipeline_state.hpp"
#include "kvf/rect.hpp"
#include "kvf/render_device_fwd.hpp"
#include "kvf/render_pass_fwd.hpp"
#include <glm/vec4.hpp>

namespace kvf::two {
class IRenderPass : public klib::Polymorphic {
  public:
	static constexpr auto samples_v = vk::SampleCountFlagBits::e1;

	[[nodiscard]] virtual auto get_render_device() const -> IRenderDevice& = 0;

	virtual void set_color_target(vk::Format format = vk::Format::eUndefined) = 0; // undefined = RGBA with swapchain color space
	virtual void set_depth_target() = 0;

	virtual void recreate(vk::SampleCountFlagBits samples = samples_v) = 0;

	[[nodiscard]] virtual auto create_pipeline(vk::PipelineLayout layout, PipelineState const& state) -> vk::UniquePipeline = 0;

	[[nodiscard]] virtual auto has_color_target() const -> bool = 0;
	[[nodiscard]] virtual auto has_resolve_target() const -> bool = 0;
	[[nodiscard]] virtual auto has_depth_target() const -> bool = 0;

	[[nodiscard]] virtual auto get_color_format() const -> vk::Format = 0;
	[[nodiscard]] virtual auto get_depth_format() const -> vk::Format = 0;
	[[nodiscard]] virtual auto get_samples() const -> vk::SampleCountFlagBits = 0;

	[[nodiscard]] virtual auto get_extent() const -> vk::Extent2D = 0;
	[[nodiscard]] virtual auto render_target() const -> RenderTarget const& = 0;

	virtual void begin_render(vk::CommandBuffer command_buffer, vk::Extent2D extent) = 0;
	[[nodiscard]] virtual auto get_command_buffer() const -> vk::CommandBuffer = 0;
	virtual auto allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> sets_layouts) -> bool = 0;
	virtual void end_render() = 0;

	[[nodiscard]] auto to_viewport(UvRect n_rect) const -> vk::Viewport;
	[[nodiscard]] auto to_scissor(UvRect n_rect) const -> vk::Rect2D;

	virtual void bind_pipeline(vk::Pipeline pipeline) const = 0;

	[[nodiscard]] virtual auto render_texture_descriptor_info(vk::Sampler sampler) const -> vk::DescriptorImageInfo = 0;

	glm::vec4 clear_color{0.0f};
	vk::ClearDepthStencilValue clear_depth{1.0f, 0};
	vk::AttachmentStoreOp depth_store_op{vk::AttachmentStoreOp::eDontCare};
};
} // namespace kvf::two
