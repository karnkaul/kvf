#pragma once
#include <kvf/buffering.hpp>
#include <kvf/color.hpp>
#include <kvf/render_device_fwd.hpp>
#include <kvf/vma.hpp>
#include <span>

namespace kvf {
struct PipelineState {
	enum : int {
		None = 0,
		AlphaBlend = 1 << 0,
		DepthTest = 1 << 1,
	};
	using Flags = int;

	[[nodiscard]] static constexpr auto default_flags() -> Flags { return AlphaBlend | DepthTest; }

	std::span<vk::VertexInputBindingDescription const> vertex_bindings;
	std::span<vk::VertexInputAttributeDescription const> vertex_attributes;
	vk::ShaderModule vertex_shader;
	vk::ShaderModule fragment_shader;

	vk::PrimitiveTopology topology{vk::PrimitiveTopology::eTriangleList};
	vk::PolygonMode polygon_mode{vk::PolygonMode::eFill};
	vk::CullModeFlags cull_mode{vk::CullModeFlagBits::eNone};
	vk::CompareOp depth_compare{vk::CompareOp::eLess};
	Flags flags{default_flags()};
};

class RenderPass {
  public:
	static constexpr auto samples_v = vk::SampleCountFlagBits::e1;

	explicit RenderPass(gsl::not_null<RenderDevice*> render_device, vk::SampleCountFlagBits samples = samples_v);

	auto set_color_target(vk::Format format = vk::Format::eUndefined) -> RenderPass&; // undefined = RGBA with swapchain color space
	auto set_depth_target() -> RenderPass&;

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
	void end_render();

	[[nodiscard]] auto viewport() const -> vk::Viewport;
	[[nodiscard]] auto scissor() const -> vk::Rect2D;
	void bind_pipeline(vk::Pipeline pipeline) const;

	Color clear_color{black_v};
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

	RenderDevice* m_device{};
	vk::SampleCountFlagBits m_samples{};

	Buffered<Framebuffer> m_framebuffers{};

	vk::CommandBuffer m_command_buffer{};
	vk::Extent2D m_extent{};

	Targets m_targets{};
	std::vector<vk::ImageMemoryBarrier2> m_barriers{};
};
} // namespace kvf
