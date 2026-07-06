#include "detail/render_pass.hpp"
#include "kvf/util.hpp"

namespace kvf::two::detail {
RenderPass::RenderPass(gsl::not_null<IRenderDevice*> render_device, vk::SampleCountFlagBits const samples) : m_device(render_device), m_samples(samples) {}

void RenderPass::set_color_target(vk::Format format) {
	if (format == vk::Format::eUndefined) {
		format = util::is_srgb(m_device->get_swapchain_color_format()) ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
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
	if (has_color_target()) { m_device->get_device().waitIdle(); }
	for (auto& framebuffer : m_framebuffers) {
		framebuffer.color = m_device->create_image(color_ici);
		if (m_samples > vk::SampleCountFlagBits::e1) { framebuffer.resolve = m_device->create_image(resolve_ici); }
	}
}

void RenderPass::set_depth_target() {
	auto const depth_ici = ImageCreateInfo{
		.format = m_device->get_optimal_depth_format(),
		.aspect = vk::ImageAspectFlagBits::eDepth,
		.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
		.samples = m_samples,
		.flags = ImageFlag::DedicatedAlloc,
		.extent = m_extent,
	};
	for (auto& framebuffer : m_framebuffers) { framebuffer.depth = m_device->create_image(depth_ici); }
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
	m_targets = {};
}

auto RenderPass::create_pipeline(vk::PipelineLayout layout, PipelineState const& state) -> vk::UniquePipeline {
	auto const format = PipelineFormat{
		.samples = m_samples,
		.color = get_color_format(),
		.depth = get_depth_format(),
	};
	return m_device->create_pipeline(layout, state, format);
}

auto RenderPass::get_color_format() const -> vk::Format {
	if (!has_color_target()) { return vk::Format::eUndefined; }
	return m_framebuffers[0].color->get_format();
}

auto RenderPass::get_depth_format() const -> vk::Format {
	if (!has_depth_target()) { return vk::Format::eUndefined; }
	return m_framebuffers[0].depth->get_format();
}

auto RenderPass::render_target() const -> RenderTarget const& {
	if (m_targets.resolve.view) { return m_targets.resolve; }
	if (m_targets.color.view) { return m_targets.color; }
	return m_targets.depth;
}

void RenderPass::begin_render(vk::CommandBuffer const command_buffer, vk::Extent2D extent) {
	if (!command_buffer || (!has_color_target() && !has_depth_target())) { return; }

	util::ensure_positive(extent);
	m_extent = extent;
	m_command_buffer = command_buffer;

	set_render_targets();

	m_barriers.clear();
	if (m_targets.color.image) {
		auto barrier = m_device->create_image_barrier();
		barrier.setImage(m_targets.color.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
		m_barriers.push_back(barrier);
	}
	if (m_targets.resolve.image) {
		KLIB_ASSERT(m_targets.color.image && !m_barriers.empty());
		auto barrier = m_barriers.back();
		barrier.setImage(m_targets.resolve.image);
		m_barriers.push_back(barrier);
	}
	if (m_targets.depth.image) {
		auto barrier = m_device->create_image_barrier(vk::ImageAspectFlagBits::eDepth);
		barrier.setImage(m_targets.depth.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
		m_barriers.push_back(barrier);
	}

	util::record_barriers(m_command_buffer, m_barriers);

	auto color_ai = vk::RenderingAttachmentInfo{};
	auto depth_ai = vk::RenderingAttachmentInfo{};

	if (m_targets.color.view) {
		auto const cc = clear_color;
		color_ai.setImageView(m_targets.color.view)
			.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearColorValue{cc.x, cc.y, cc.z, cc.w});
	}
	if (m_targets.resolve.view) {
		color_ai.setResolveImageView(m_targets.resolve.view)
			.setResolveImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setResolveMode(vk::ResolveModeFlagBits::eAverage);
	}
	if (m_targets.depth.view) {
		depth_ai.setImageView(m_targets.depth.view)
			.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(depth_store_op)
			.setClearValue(clear_depth);
	}

	auto ri = vk::RenderingInfo{};
	if (depth_ai.imageView) { ri.setPDepthAttachment(&depth_ai).setRenderArea(vk::Rect2D{{}, m_targets.depth.extent}); }
	if (color_ai.imageView) { ri.setColorAttachments(color_ai).setLayerCount(1).setRenderArea(vk::Rect2D{{}, m_targets.color.extent}); }
	m_command_buffer.beginRendering(ri);
}

auto RenderPass::allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> sets_layouts) -> bool {
	return m_device->get_descriptor_allocator().allocate_next(out_sets, sets_layouts);
}

void RenderPass::end_render() {
	if (m_command_buffer) { m_command_buffer.endRendering(); }

	m_barriers.clear();
	if (m_targets.color.image) {
		auto barrier = m_device->create_image_barrier();
		barrier.setImage(m_targets.color.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eTransferRead)
			.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eTransfer)
			.setOldLayout(vk::ImageLayout::eAttachmentOptimal)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		m_barriers.push_back(barrier);
	}
	if (m_targets.resolve.image) {
		KLIB_ASSERT(m_targets.color.image && !m_barriers.empty());
		auto barrier = m_barriers.back();
		barrier.setImage(m_targets.resolve.image);
		m_barriers.push_back(barrier);
	}
	if (m_targets.depth.image && depth_store_op == vk::AttachmentStoreOp::eStore) {
		auto barrier = m_device->create_image_barrier(vk::ImageAspectFlagBits::eDepth);
		barrier.setImage(m_targets.depth.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setOldLayout(vk::ImageLayout::eAttachmentOptimal)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		m_barriers.push_back(barrier);
	}
	if (m_command_buffer) { util::record_barriers(m_command_buffer, m_barriers); }

	m_command_buffer = vk::CommandBuffer{};
}

void RenderPass::bind_pipeline(vk::Pipeline const pipeline) const {
	if (!m_command_buffer) { return; }
	m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	m_command_buffer.setViewport(0, to_viewport(uv_rect_v));
	m_command_buffer.setScissor(0, to_scissor(uv_rect_v));
}

auto RenderPass::render_texture_descriptor_info(vk::Sampler const sampler) const -> vk::DescriptorImageInfo {
	auto const& rt = render_target();
	if (!rt.view) { return {}; }
	auto ret = vk::DescriptorImageInfo{};
	ret.setImageView(rt.view).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal).setSampler(sampler);
	return ret;
}

void RenderPass::set_render_targets() {
	m_targets = {};

	auto& framebuffer = m_framebuffers.at(std::size_t(m_device->get_frame_index()));

	if (framebuffer.color) {
		framebuffer.color->resize(m_extent);
		m_targets.color = framebuffer.color->render_target();

		if (framebuffer.resolve) {
			framebuffer.resolve->resize(m_extent);
			m_targets.resolve = framebuffer.resolve->render_target();
		}
	}

	if (framebuffer.depth) {
		framebuffer.depth->resize(m_extent);
		m_targets.depth = framebuffer.depth->render_target();
	}
}
} // namespace kvf::two::detail

namespace kvf::two {
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
} // namespace kvf::two
