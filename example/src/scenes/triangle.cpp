#include <imgui.h>
#include <kvf/error.hpp>
#include <kvf/util.hpp>
#include <scenes/triangle.hpp>
#include <shader_loader.hpp>

namespace kvf::example {
Triangle::Triangle(gsl::not_null<RenderDevice*> device, std::string_view assets_dir)
	: Scene(device, assets_dir), m_color_pass(device, vk::SampleCountFlagBits::e2) {
	m_color_pass.set_color_target().set_depth_target();
	m_color_pass.clear_color = vk::ClearColorValue{std::array{0.05f, 0.05f, 0.05f, 1.0f}};
	create_pipeline();
}

void Triangle::update(vk::CommandBuffer const command_buffer) {
	ImGui::SetNextWindowSize({150.0f, 80.0f}, ImGuiCond_Once);
	if (ImGui::Begin("Controls")) { draw_controls(); }
	ImGui::End();

	auto const extent = kvf::util::scale_extent(get_render_device().get_framebuffer_extent(), m_framebuffer_scale);
	m_color_pass.begin_render(command_buffer, extent);

	m_color_pass.bind_pipeline(*m_pipeline);
	command_buffer.draw(3, 1, 0, 0);

	m_color_pass.end_render();
}

auto Triangle::get_render_target() const -> RenderTarget { return m_color_pass.render_target(); }

void Triangle::create_pipeline() {
	auto loader = ShaderLoader{get_render_device().get_device(), get_assets_dir()};
	auto const vertex_shader = loader.load("triangle.vert");
	auto const fragment_shader = loader.load("triangle.frag");

	m_pipeline_layout = get_render_device().get_device().createPipelineLayoutUnique({});
	auto const pipeline_state = kvf::PipelineState{
		.vertex_bindings = {},
		.vertex_attributes = {},
		.vertex_shader = *vertex_shader,
		.fragment_shader = *fragment_shader,
	};
	m_pipeline = m_color_pass.create_pipeline(*m_pipeline_layout, pipeline_state);
	if (!m_pipeline) { throw Error{"Failed to create Vulkan Pipeline"}; }
}

void Triangle::draw_controls() {
	ImGui::TextUnformatted("framebuffer scale");
	ImGui::SliderFloat("##fb_scale", &m_framebuffer_scale, 0.25f, 2.0f, "%.2f");
}
} // namespace kvf::example
