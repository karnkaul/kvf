#pragma once
#include <kvf/vma.hpp>
#include <scene.hpp>

namespace kvf::example {
class Standalone : public Scene {
  public:
	explicit Standalone(gsl::not_null<RenderDevice*> device, std::string_view assets_dir);

  private:
	[[nodiscard]] auto get_render_filter() const -> vk::Filter final;
	[[nodiscard]] auto get_render_target() const -> RenderTarget final;

	vma::Image m_image;
};
} // namespace kvf::example
