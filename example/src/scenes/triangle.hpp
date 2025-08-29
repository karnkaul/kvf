#pragma once
#include <kvf/render_pass.hpp>
#include <scene.hpp>

namespace kvf::example {
class Triangle : public Scene {
  public:
	explicit Triangle(gsl::not_null<RenderDevice*> device, std::string_view assets_dir);

  private:
	void on_key(KeyInput const& ki) final;
	void update(vk::CommandBuffer command_buffer) final;
	[[nodiscard]] auto get_render_target() const -> RenderTarget final;

	void create_pipeline();

	void recreate(vk::SampleCountFlagBits samples);

	void draw_controls();

	kvf::RenderPass m_color_pass;
	float m_framebuffer_scale{2.0f};

	vk::UniquePipelineLayout m_pipeline_layout{};
	vk::UniquePipeline m_pipeline{};
};
} // namespace kvf::example
