#include "scenes/standalone.hpp"
#include "kvf/color_bitmap.hpp"
#include "kvf/error.hpp"
#include "kvf/util.hpp"

namespace kvf::example {
Standalone::Standalone(gsl::not_null<two::IRenderDevice*> device, std::string_view assets_dir) : Scene(device, assets_dir) {
	auto pixels = ColorBitmap{glm::ivec2{2, 2}};
	pixels[0, 0] = red_v;
	pixels[0, 1] = green_v;
	pixels[1, 0] = blue_v;
	pixels[1, 1] = yellow_v;
	auto const bitmap = pixels.bitmap();
	auto const ici = two::ImageCreateInfo{
		.format = vk::Format::eR8G8B8A8Unorm,
		.extent = util::to_vk_extent(bitmap.size),
	};
	m_image = device->create_image(ici);
	if (!m_image->resize_and_overwrite(bitmap)) { throw Error{"Failed to write to Image"}; }
}

auto Standalone::get_render_filter() const -> vk::Filter { return vk::Filter::eNearest; }

auto Standalone::get_render_target() const -> RenderTarget { return m_image->render_target(); }
} // namespace kvf::example
