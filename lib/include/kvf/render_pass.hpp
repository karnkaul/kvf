#pragma once
#include "klib/base_types.hpp"
#include "kvf/pipeline_state.hpp"
#include "kvf/rect.hpp"
#include "kvf/render_device_fwd.hpp"
#include "kvf/render_pass_fwd.hpp"
#include "kvf/render_target.hpp"
#include <glm/vec4.hpp>

namespace kvf {
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
} // namespace kvf
