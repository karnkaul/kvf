#pragma once
#include <klib/c_string.hpp>
#include <kvf/vma.hpp>
#include <scene.hpp>

namespace kvf::example {
class ImageViewer : public Scene {
  public:
	explicit ImageViewer(gsl::not_null<RenderDevice*> device, std::string_view assets_dir);

  private:
	void on_drop(std::span<char const* const> paths) final;

	[[nodiscard]] auto get_render_target() const -> RenderTarget final;

	void resize_window();
	void try_load(klib::CString path);

	vma::Image m_image;
};
} // namespace kvf::example
