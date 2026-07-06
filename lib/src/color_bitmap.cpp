#include "kvf/color_bitmap.hpp"
#include <glm/gtc/color_space.hpp>
#include <glm/mat4x4.hpp>
#include <utility>

namespace kvf {
auto Color::linear_to_srgb(glm::vec4 const& channels) -> glm::vec4 { return glm::convertLinearToSRGB(channels); }
auto Color::srgb_to_linear(glm::vec4 const& channels) -> glm::vec4 { return glm::convertSRGBToLinear(channels); }

void ColorBitmap::resize(glm::ivec2 size) {
	if (size.x < 0 || size.y < 0) { return; }
	m_size = size;
	m_bitmap.resize(std::size_t(m_size.x * m_size.y));
}

auto ColorBitmap::at(int const x, int const y) const -> Color const& {
	auto const index = (y * m_size.x) + x;
	return m_bitmap.at(std::size_t(index));
}

auto ColorBitmap::at(int const x, int const y) -> Color& {
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
	return const_cast<Color&>(std::as_const(*this).at(x, y));
}

auto ColorBitmap::bitmap() const -> Bitmap {
	static_assert(sizeof(Color) == Bitmap::channels_v);
	void const* first = m_bitmap.data();
	return Bitmap{
		.bytes = std::span{static_cast<std::byte const*>(first), sizeof(Color) * m_bitmap.size()},
		.size = m_size,
	};
}
} // namespace kvf
