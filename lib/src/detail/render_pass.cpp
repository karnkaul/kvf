#include "detail/render_pass.hpp"
#include "kvf/render_device.hpp"
#include "kvf/util.hpp"
#include <array>

namespace kvf::detail {
RenderPass::RenderPass(gsl::not_null<IRenderDevice*> render_device, vk::SampleCountFlagBits const samples)
	: m_render_device(render_device), m_samples(samples) {}

void RenderPass::set_color_target(vk::Format format) {
	if (format == vk::Format::eUndefined) {
		format = util::is_srgb(m_render_device->get_swapchain_color_format()) ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
	}

	auto const color_ici = ImageCreateInfo{
		.format = format,
		.aspect = vk::ImageAspectFlagBits::eColor,
		.usage = vk::ImageUsageFlagBits::eColorAttachment,
		.samples = m_samples,
		.flags = ImageFlag::DedicatedAlloc,
		.extent = m_extent,
	};
	auto const resolve_ici = [&] {
		auto ret = color_ici;
		ret.samples = vk::SampleCountFlagBits::e1;
		ret.extent = m_extent;
		return ret;
	}();
	if (has_color_target()) { m_render_device->get_device().waitIdle(); }
	for (auto& framebuffer : m_framebuffers) {
		framebuffer.color.emplace(m_render_device, color_ici);
		if (m_samples > vk::SampleCountFlagBits::e1) { framebuffer.resolve.emplace(m_render_device, resolve_ici); }
	}
}

void RenderPass::set_depth_target() {
	auto const depth_ici = ImageCreateInfo{
		.format = m_render_device->get_optimal_depth_format(),
		.aspect = vk::ImageAspectFlagBits::eDepth,
		.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
		.samples = m_samples,
		.flags = ImageFlag::DedicatedAlloc,
		.extent = m_extent,
	};
	for (auto& framebuffer : m_framebuffers) { framebuffer.depth.emplace(m_render_device, depth_ici); }
}

void RenderPass::recreate(vk::SampleCountFlagBits const samples) {
	auto const had_color = has_color_target();
	auto const had_depth = has_depth_target();
	m_samples = samples;
	if (had_color) { set_color_target(get_color_format()); }
	if (had_depth) { set_depth_target(); }
	if (m_samples == vk::SampleCountFlagBits::e1) {
		for (auto& framebuffer : m_framebuffers) { framebuffer.resolve.reset(); }
	}
}

auto RenderPass::create_graphics_pipeline(vk::PipelineLayout const layout, PipelineState const& state) -> vk::UniquePipeline {
	auto const format = PipelineFormat{
		.samples = m_samples,
		.color = get_color_format(),
		.depth = get_depth_format(),
	};
	return m_render_device->create_pipeline(layout, state, format);
}

auto RenderPass::get_color_format() const -> vk::Format {
	if (!has_color_target()) { return vk::Format::eUndefined; }
	return m_framebuffers[0].color->get_format();
}

auto RenderPass::get_depth_format() const -> vk::Format {
	if (!has_depth_target()) { return vk::Format::eUndefined; }
	return m_framebuffers[0].depth->get_format();
}

void RenderPass::begin_render(vk::CommandBuffer const command_buffer, vk::Extent2D extent) {
	if (!command_buffer || (!has_color_target() && !has_depth_target())) { return; }

	util::ensure_positive(extent);
	m_extent = extent;
	m_command_buffer = command_buffer;

	auto& framebuffer = m_framebuffers.at(std::size_t(m_render_device->get_frame_index()));
	prep_for_render(framebuffer);

	m_barriers.clear();
	if (framebuffer.color) { m_barriers.push_back(framebuffer.color->get_pre_render_barrier()); }
	if (framebuffer.resolve) { m_barriers.push_back(framebuffer.resolve->get_pre_render_barrier()); }
	if (framebuffer.depth) { m_barriers.push_back(framebuffer.depth->get_pre_render_barrier()); }
	util::record_barriers(m_command_buffer, m_barriers);

	auto color_ai = vk::RenderingAttachmentInfo{};
	auto depth_ai = vk::RenderingAttachmentInfo{};

	if (framebuffer.color) {
		auto const cc = clear_color;
		color_ai.setImageView(framebuffer.color->get_image_view())
			.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearColorValue{cc.x, cc.y, cc.z, cc.w});
	}
	if (framebuffer.resolve) {
		color_ai.setResolveImageView(framebuffer.resolve->get_image_view())
			.setResolveImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setResolveMode(vk::ResolveModeFlagBits::eAverage);
	}
	if (framebuffer.depth) {
		depth_ai.setImageView(framebuffer.depth->get_image_view())
			.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(depth_store_op)
			.setClearValue(clear_depth);
	}

	auto rendering_info = vk::RenderingInfo{};
	if (framebuffer.depth) { rendering_info.setPDepthAttachment(&depth_ai).setRenderArea(vk::Rect2D{{}, m_extent}); }
	if (framebuffer.color) { rendering_info.setColorAttachments(color_ai).setLayerCount(1).setRenderArea(vk::Rect2D{{}, m_extent}); }
	m_command_buffer.beginRendering(rendering_info);
	if ((m_render_device->get_flags() & RenderDeviceFlag::ShaderObjectFeature) == RenderDeviceFlag::ShaderObjectFeature) {
		m_command_buffer.setRasterizationSamplesEXT(m_samples);
		m_command_buffer.setSampleMaskEXT(m_samples, vk::SampleMask{0xffffffff});
	}
}

auto RenderPass::allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> sets_layouts) -> bool {
	return m_render_device->get_descriptor_allocator().allocate_next(out_sets, sets_layouts);
}

void RenderPass::end_render() {
	if (!m_command_buffer) { return; }

	m_command_buffer.endRendering();

	auto& framebuffer = m_framebuffers.at(std::size_t(m_render_device->get_frame_index()));

	m_barriers.clear();
	if (framebuffer.color) { m_barriers.push_back(framebuffer.color->get_post_render_barrier()); }
	if (framebuffer.resolve) { m_barriers.push_back(framebuffer.resolve->get_post_render_barrier()); }
	if (framebuffer.depth && depth_store_op == vk::AttachmentStoreOp::eStore) { m_barriers.push_back(framebuffer.depth->get_post_render_barrier()); }
	util::record_barriers(m_command_buffer, m_barriers);

	m_command_buffer = vk::CommandBuffer{};
	m_rendered_index = m_render_device->get_frame_index();
	m_render_target = framebuffer.render_image()->render_target();
}

void RenderPass::bind_graphics_pipeline(vk::Pipeline const pipeline) const {
	if (!m_command_buffer) { return; }
	m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	m_command_buffer.setViewport(0, to_viewport(uv_rect_v));
	m_command_buffer.setScissor(0, to_scissor(uv_rect_v));
}

void RenderPass::bind_graphics_shader(IGraphicsShader const& shader) const {
	if (!m_command_buffer) { return; }

	static constexpr auto stages_v = std::array{
		vk::ShaderStageFlagBits::eVertex,
		vk::ShaderStageFlagBits::eFragment,
	};
	auto const shaders = std::array{
		shader.get_vertex(),
		shader.get_fragment(),
	};
	m_command_buffer.bindShadersEXT(stages_v, shaders);
	m_command_buffer.setRasterizerDiscardEnable(vk::False);
	m_command_buffer.setDepthBoundsTestEnable(vk::False);
	m_command_buffer.setDepthBiasEnable(vk::False);
	m_command_buffer.setStencilTestEnable(vk::False);
	m_command_buffer.setLogicOpEnableEXT(vk::False);
	static constexpr auto color_blend_eq_v = [] {
		auto ret = vk::ColorBlendEquationEXT{};
		ret.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
			.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
			.setColorBlendOp(vk::BlendOp::eAdd)
			.setSrcAlphaBlendFactor(vk::BlendFactor::eZero)
			.setDstAlphaBlendFactor(vk::BlendFactor::eOne)
			.setAlphaBlendOp(vk::BlendOp::eAdd);
		return ret;
	}();
	m_command_buffer.setColorBlendEquationEXT(0, color_blend_eq_v);
	m_command_buffer.setPrimitiveRestartEnable(vk::False);
	auto const ccf = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	m_command_buffer.setColorWriteMaskEXT(0, ccf);
	m_command_buffer.setAlphaToCoverageEnableEXT(vk::False);
	m_command_buffer.setFrontFace(vk::FrontFace::eCounterClockwise);

	m_command_buffer.setCullMode(vk::CullModeFlagBits::eNone);
	m_command_buffer.setDepthTestEnable(vk::False);
	m_command_buffer.setDepthWriteEnable(vk::False);
	m_command_buffer.setPolygonModeEXT(vk::PolygonMode::eFill);
	m_command_buffer.setColorBlendEnableEXT(0, vk::True);

	auto const input = shader.get_input();
	m_command_buffer.setVertexInputEXT(input.bindings, input.attributes);
	m_command_buffer.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);

	m_command_buffer.setViewportWithCount(to_viewport(uv_rect_v));
	m_command_buffer.setScissorWithCount(to_scissor(uv_rect_v));
	m_command_buffer.setRasterizationSamplesEXT(m_samples);
	m_command_buffer.setSampleMaskEXT(m_samples, vk::SampleMask{0xffffffff});
}

auto RenderPass::render_texture_descriptor_info(vk::Sampler const sampler) const -> std::optional<vk::DescriptorImageInfo> {
	auto const render_image = get_rendered_image();
	if (!render_image) { return {}; }
	auto ret = vk::DescriptorImageInfo{};
	ret.setImageView(render_image->get_image_view()).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal).setSampler(sampler);
	return ret;
}

auto RenderPass::copy_render_texture(vk::Extent2D const custom_extent) const -> std::optional<ColorBitmap> {
	auto const render_image = get_rendered_image();
	if (!render_image) { return {}; }
	return render_image->copy_to_bitmap(custom_extent);
}

auto RenderPass::get_rendered_image() const -> klib::Ptr<IRenderImage const> {
	if (!m_rendered_index) { return {}; }
	auto const& framebuffer = m_framebuffers.at(std::size_t(*m_rendered_index));
	return framebuffer.render_image();
}

void RenderPass::prep_for_render(Framebuffer& framebuffer) {
	if (framebuffer.color) {
		framebuffer.color->resize(m_extent);
		if (framebuffer.resolve) { framebuffer.resolve->resize(m_extent); }
	}
	if (framebuffer.depth) { framebuffer.depth->resize(m_extent); }
}

auto RenderPass::Framebuffer::render_image() const -> klib::Ptr<IRenderImage const> {
	if (resolve) { return &*resolve; }
	if (color) { return &*color; }
	if (depth) { return &*depth; }
	return nullptr;
}

} // namespace kvf::detail

namespace kvf {

auto IRenderPass::create(gsl::not_null<IRenderDevice*> render_device, vk::SampleCountFlagBits const samples) -> std::unique_ptr<IRenderPass> {
	return std::make_unique<detail::RenderPass>(render_device, samples);
}

auto IRenderPass::to_viewport(UvRect n_rect) const -> vk::Viewport {
	if (!util::is_norm(n_rect)) { n_rect = uv_rect_v; }
	auto const fb_size = util::to_glm_vec(get_extent());
	auto const rect = UvRect{.lt = n_rect.lt * fb_size, .rb = n_rect.rb * fb_size};
	auto const vp_size = rect.size();
	return vk::Viewport{rect.lt.x, rect.rb.y, vp_size.x, -vp_size.y};
}

auto IRenderPass::to_scissor(UvRect n_rect) const -> vk::Rect2D {
	if (!util::is_norm(n_rect)) { n_rect = uv_rect_v; }
	auto const fb_size = kvf::util::to_glm_vec(get_extent());
	auto const rect = kvf::UvRect{.lt = n_rect.lt * fb_size, .rb = n_rect.rb * fb_size};
	auto const offset = glm::ivec2{rect.lt};
	auto const extent = glm::uvec2{rect.size()};
	return vk::Rect2D{vk::Offset2D{offset.x, offset.y}, vk::Extent2D{extent.x, extent.y}};
}
} // namespace kvf
