#include <kvf/color_bitmap.hpp>
#include <kvf/error.hpp>
#include <kvf/util.hpp>
#include <scenes/standalone.hpp>

namespace kvf::example {
Standalone::Standalone(gsl::not_null<RenderDevice*> device, std::string_view assets_dir) : Scene(device, assets_dir) {
	auto pixels = ColorBitmap{glm::ivec2{2, 2}};
	pixels[0, 0] = red_v;
	pixels[0, 1] = green_v;
	pixels[1, 0] = blue_v;
	pixels[1, 1] = yellow_v;
	auto const bitmap = pixels.bitmap();
	auto const ici = vma::ImageCreateInfo{
		.format = vk::Format::eR8G8B8A8Unorm,
	};
	m_image = vma::Image{device, ici, util::to_vk_extent(bitmap.size)};
	if (!util::write_to(m_image, bitmap)) { throw Error{"Failed to write to Image"}; }
}

auto Standalone::get_render_filter() const -> vk::Filter { return vk::Filter::eNearest; }

auto Standalone::get_render_target() const -> RenderTarget { return m_image.render_target(); }
} // namespace kvf::example
