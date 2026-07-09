#pragma once
#include <glm/vec2.hpp>
#include <bit>
#include <cstddef>
#include <span>

namespace kvf {
struct Bitmap {
	static constexpr std::uint32_t channels_v{4};

	std::span<std::byte const> bytes{};
	glm::ivec2 size{};
};

struct BytePixel {
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
	std::byte bytes[4];
};

inline constexpr auto white_pixel_v = std::bit_cast<BytePixel>(0xffffffff);
inline constexpr auto white_bitmap_v = Bitmap{.bytes = white_pixel_v.bytes, .size = {1, 1}};
} // namespace kvf
