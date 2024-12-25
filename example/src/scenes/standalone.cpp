#include <kvf/error.hpp>
#include <kvf/util.hpp>
#include <scenes/standalone.hpp>

namespace kvf::example {
Standalone::Standalone(gsl::not_null<RenderDevice*> device, std::string_view assets_dir) : Scene(device, assets_dir), m_blocker(device->get_device()) {
	auto const image_bytes = std::array{
		std::byte{0xff}, std::byte{},	  std::byte{},	   std::byte{0xff},

		std::byte{},	 std::byte{0xff}, std::byte{},	   std::byte{0xff},

		std::byte{},	 std::byte{},	  std::byte{0xff}, std::byte{0xff},

		std::byte{0xff}, std::byte{0xff}, std::byte{},	   std::byte{0xff},
	};
	auto const bitmap = RgbaBitmap{.bytes = image_bytes, .extent = {2, 2}};
	auto const ici = vma::ImageCreateInfo{
		.format = vk::Format::eR8G8B8A8Unorm,
	};
	m_image = vma::Image{device, ici, bitmap.extent};
	if (!util::write_to(m_image, bitmap)) { throw Error{"Failed to write to Image"}; }
}

auto Standalone::get_render_filter() const -> vk::Filter { return vk::Filter::eNearest; }

auto Standalone::get_render_target() const -> RenderTarget { return m_image.render_target(); }
} // namespace kvf::example
