#pragma once
#include "kvf/image.hpp"
#include "kvf/render_device.hpp"
#include "scene.hpp"

namespace kvf::example {
class Standalone : public Scene {
  public:
	explicit Standalone(gsl::not_null<IRenderDevice*> device, std::string_view assets_dir);

  private:
	[[nodiscard]] auto get_render_filter() const -> vk::Filter final;
	[[nodiscard]] auto get_render_target() const -> RenderTarget final;

	std::unique_ptr<IImage> m_image;
};
} // namespace kvf::example
