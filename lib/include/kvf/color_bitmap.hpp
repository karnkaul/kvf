#pragma once
#include <kvf/bitmap.hpp>
#include <kvf/color.hpp>
#include <vector>

namespace kvf {
class ColorBitmap {
  public:
	ColorBitmap() = default;

	explicit ColorBitmap(glm::ivec2 size) { resize(size); }

	void resize(glm::ivec2 size);

	[[nodiscard]] auto at(int x, int y) const -> Color const&;
	[[nodiscard]] auto at(int x, int y) -> Color&;

	auto operator[](int x, int y) const -> Color const& { return at(x, y); }
	auto operator[](int x, int y) -> Color& { return at(x, y); }

	[[nodiscard]] auto bitmap() const -> Bitmap;

  private:
	std::vector<Color> m_bitmap{};
	glm::ivec2 m_size{};
};
} // namespace kvf
