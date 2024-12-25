#pragma once
#include <kvf/render_pass.hpp>
#include <scene.hpp>

namespace kvf::example {
class Triangle : public Scene {
  public:
	explicit Triangle(gsl::not_null<RenderDevice*> device, std::string_view assets_dir);

  private:
	void update(vk::CommandBuffer command_buffer) final;
	[[nodiscard]] auto get_render_target() const -> RenderTarget final;

	void create_pipeline();

	kvf::RenderPass m_color_pass;

	vk::UniquePipelineLayout m_pipeline_layout{};
	vk::UniquePipeline m_pipeline{};
};
} // namespace kvf::example
