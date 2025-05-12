#pragma once
#include <glm/vec4.hpp>
#include <klib/byte_cast.hpp>
#include <kvf/bitmap.hpp>
#include <cstdint>

namespace kvf {
using GlmColor = glm::tvec4<std::uint8_t>;

class Color : public GlmColor {
  public:
	static constexpr std::uint8_t channel_max_v{0xff};

	static constexpr auto to_f32(std::uint8_t const channel) -> float { return float(channel) / float(channel_max_v); }
	static constexpr auto to_u8(float const norm) -> std::uint8_t { return std::uint8_t(norm * float(channel_max_v)); }

	static constexpr auto red(std::uint32_t const mask) -> std::uint8_t { return std::uint8_t((mask >> (3 * 8)) & 0xff); }
	static constexpr auto green(std::uint32_t const mask) -> std::uint8_t { return std::uint8_t((mask >> (2 * 8)) & 0xff); }
	static constexpr auto blue(std::uint32_t const mask) -> std::uint8_t { return std::uint8_t((mask >> (1 * 8)) & 0xff); }
	static constexpr auto alpha(std::uint32_t const mask) -> std::uint8_t { return std::uint8_t((mask >> (0 * 8)) & 0xff); }

	Color() = default;

	constexpr Color(GlmColor const c) : GlmColor(c) {}
	constexpr Color(glm::vec4 const& norm) : GlmColor(to_u8(norm.x), to_u8(norm.y), to_u8(norm.z), to_u8(norm.w)) {}
	explicit constexpr Color(std::uint32_t const mask) : GlmColor(red(mask), green(mask), blue(mask), alpha(mask)) {}

	[[nodiscard]] constexpr auto to_u32() const -> std::uint32_t {
		return (std::uint32_t(x) << (3 * 8)) | (std::uint32_t(y) << (2 * 8)) | (std::uint32_t(z) << 8) | std::uint32_t(w);
	}

	[[nodiscard]] constexpr auto to_vec4() const -> glm::vec4 { return {to_f32(x), to_f32(y), to_f32(z), to_f32(w)}; }

	[[nodiscard]] static auto linear_to_srgb(glm::vec4 const& channels) -> glm::vec4;
	[[nodiscard]] static auto srgb_to_linear(glm::vec4 const& channels) -> glm::vec4;

	[[nodiscard]] auto to_srgb() const -> glm::vec4 { return linear_to_srgb(to_vec4()); }
	[[nodiscard]] auto to_linear() const -> glm::vec4 { return srgb_to_linear(to_vec4()); }
};

constexpr auto black_v = Color{0x0000000ff};
constexpr auto white_v = Color{0xffffffff};
constexpr auto red_v = Color{0xff0000ff};
constexpr auto green_v = Color{0x00ff00ff};
constexpr auto blue_v = Color{0x0000ffff};
constexpr auto cyan_v = Color{green_v | blue_v};
constexpr auto yellow_v = Color{red_v | green_v};
constexpr auto magenta_v = Color{blue_v | red_v};

template <Color C>
inline constexpr auto pixel_bytes_v = klib::byte_cast(C);

template <Color C>
inline constexpr auto pixel_bitmap_v = Bitmap{.bytes = pixel_bytes_v<C>, .size = {1, 1}};
} // namespace kvf
