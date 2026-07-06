#include "kvf/image_bitmap.hpp"
#include "kvf/is_positive.hpp"
#include <stb/stb_image.h>

namespace kvf {
void ImageBitmap::Deleter::operator()(Bitmap const& bitmap) const noexcept {
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
	stbi_image_free(const_cast<std::byte*>(bitmap.bytes.data()));
}

ImageBitmap::ImageBitmap(std::span<std::byte const> compressed) { decompress(compressed); }

auto ImageBitmap::decompress(std::span<std::byte const> compressed) -> bool {
	auto const* ptr = static_cast<void const*>(compressed.data());
	auto size = glm::ivec2{};
	auto in_channels = int{};
	void* result = stbi_load_from_memory(static_cast<stbi_uc const*>(ptr), int(compressed.size()), &size.x, &size.y, &in_channels, int(channels_v));
	if (result == nullptr || !is_positive(size)) { return false; }

	m_bitmap = Bitmap{
		.bytes = std::span{static_cast<std::byte const*>(result), std::size_t(size.x * size.y * int(channels_v))},
		.size = size,
	};

	return true;
}
} // namespace kvf
