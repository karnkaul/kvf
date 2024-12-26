#include <klib/assert.hpp>
#include <kvf/error.hpp>
#include <kvf/util.hpp>
#include <log.hpp>
#include <scenes/image_viewer.hpp>
#include <algorithm>
#include <filesystem>

namespace kvf::example {
namespace {
namespace fs = std::filesystem;

[[nodiscard]] auto find_image_file(std::span<char const* const> paths) -> klib::CString {
	static constexpr auto extensions_v = std::array{
		".jpg", ".jpeg", ".bmp", ".png", ".ppm",
	};
	for (auto const* path : paths) {
		auto const extension = fs::path{path}.extension().string();
		if (std::ranges::find(extensions_v, extension) == extensions_v.end()) { continue; }
		return path;
	}
	return {};
}
} // namespace

ImageViewer::ImageViewer(gsl::not_null<RenderDevice*> device, std::string_view assets_dir) : Scene(device, assets_dir) {
	auto const ici = vma::ImageCreateInfo{.format = vk::Format::eR8G8B8A8Srgb};
	static constexpr auto image_bytes_v = std::array{std::byte{}, std::byte{}, std::byte{}, std::byte{0xff}};
	auto const bitmap = Bitmap{.bytes = image_bytes_v, .size = {1, 1}};
	m_image = vma::Image{device, ici, util::to_vk_extent(bitmap.size)};
	if (!util::write_to(m_image, bitmap)) { throw Error{"Failed to write to Image"}; }
	resize_window();
}

auto ImageViewer::get_render_target() const -> RenderTarget { return m_image.render_target(); }

void ImageViewer::on_drop(std::span<char const* const> paths) {
	auto const path = find_image_file(paths);
	if (path.as_view().empty()) { return; }
	try_load(path);
}

void ImageViewer::resize_window() {
	auto extent = m_image.get_extent();
	KLIB_ASSERT(extent.height > 0);
	auto const aspect_ratio = float(extent.width) / float(extent.height);

	static constexpr std::uint32_t min_height_v{400};
	static constexpr std::uint32_t max_height_v{900};
	extent.height = std::clamp(extent.height, min_height_v, max_height_v);
	extent.width = std::uint32_t(float(extent.height) * aspect_ratio);
	glfwSetWindowSize(get_device().get_window(), int(extent.width), int(extent.height));
	glfwSetWindowAspectRatio(get_device().get_window(), int(extent.width), int(extent.height));
}

void ImageViewer::try_load(klib::CString const path) {
	auto const filename = [path] { return fs::path{path.as_view()}.filename().generic_string(); };
	auto bytes = std::vector<std::byte>{};
	if (util::bytes_from_file(bytes, path) != IoResult::Success) {
		open_error_modal(std::format("Failed to load image file: {}", filename()));
		return;
	}
	auto const rgba_image = RgbaImage{bytes};
	if (!rgba_image.is_loaded()) {
		open_error_modal(std::format("Failed to create RgbaImage from file: {}", filename()));
		return;
	}
	get_device().get_device().waitIdle();
	if (!util::write_to(m_image, rgba_image.bitmap())) {
		open_error_modal(std::format("Failed to write to Vulkan Image: {}", filename()));
		return;
	}
	resize_window();
}
} // namespace kvf::example
