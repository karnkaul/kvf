#pragma once
#include <glm/vec2.hpp>
#include <kvf/render_pass.hpp>
#include <scene.hpp>

namespace kvf::example {
class Sprite : public Scene {
  public:
	explicit Sprite(gsl::not_null<RenderDevice*> device, std::string_view assets_dir);

  private:
	void update(vk::CommandBuffer command_buffer) final;
	[[nodiscard]] auto get_render_target() const -> RenderTarget final;

	void create_set_layouts();
	void create_pipeline_layout();
	void create_pipeline();

	void create_descriptor_pools();

	void write_vbo();
	void write_descriptor_sets(std::span<vk::DescriptorSet const, 2> sets, glm::vec2 extent);

	kvf::RenderPass m_color_pass;

	std::array<vk::UniqueDescriptorSetLayout, 2> m_set_layout_storage{};
	std::array<vk::DescriptorSetLayout, 2> m_set_layouts{};
	vk::UniquePipelineLayout m_pipeline_layout{};
	vk::UniquePipeline m_pipeline{};

	Buffered<vk::UniqueDescriptorPool> m_descriptor_pools{};

	vma::Buffer m_vbo;
	vk::DeviceSize m_index_offset{};

	vma::Buffer m_ubo;
	vma::Buffer m_ssbo;
};
} // namespace kvf::example
