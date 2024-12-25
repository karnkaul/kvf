#include <kvf/util.hpp>
#include <scenes/triangle.hpp>
#include <shader_loader.hpp>

namespace kvf::example {
Triangle::Triangle(gsl::not_null<RenderDevice*> device, std::string_view assets_dir)
	: Scene(device, assets_dir), m_color_pass(device, vk::SampleCountFlagBits::e2), m_blocker(device->get_device()) {
	m_color_pass.set_color_target().set_depth_target();
	create_pipeline();
}

void Triangle::update(vk::CommandBuffer const command_buffer) {
	static constexpr auto upscale_v = 2.0f;
	auto const framebuffer_extent = kvf::util::scale_extent(get_device().get_framebuffer_extent(), upscale_v);
	m_color_pass.begin_render(command_buffer, framebuffer_extent);

	m_color_pass.bind_pipeline(*m_pipeline);
	command_buffer.draw(3, 1, 0, 0);

	m_color_pass.end_render();
}

auto Triangle::get_render_target() const -> RenderTarget { return m_color_pass.render_target(); }

void Triangle::create_pipeline() {
	auto loader = ShaderLoader{get_device().get_device(), get_assets_dir()};
	auto const vertex_shader = loader.load("shader.vert");
	auto const fragment_shader = loader.load("shader.frag");

	m_pipeline_layout = get_device().get_device().createPipelineLayoutUnique({});
	auto const pipeline_state = kvf::PipelineState{
		.vertex_attributes = {},
		.vertex_bindings = {},
		.vertex_shader = *vertex_shader,
		.fragment_shader = *fragment_shader,
	};
	m_pipeline = m_color_pass.create_pipeline(*m_pipeline_layout, pipeline_state);
	if (!m_pipeline) { throw std::runtime_error{"Failed to create Vulkan Pipeline"}; }
}
} // namespace kvf::example
