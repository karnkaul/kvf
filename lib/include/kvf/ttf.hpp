#pragma once
#include <klib/c_string.hpp>
#include <kvf/color_bitmap.hpp>
#include <kvf/rect.hpp>
#include <cstddef>
#include <cstdint>
#include <gsl/pointers>
#include <memory>
#include <span>
#include <string_view>

namespace kvf::ttf {
// NOLINTNEXTLINE(performance-enum-size)
enum struct Codepoint : std::uint32_t {
	Tofu = 0,
	Space = 32,
	AsciiFirst = Space,
	AsciiLast = 126,
};

enum struct GlyphIndex : std::uint32_t {};

struct Slot {
	glm::ivec2 size{};
	glm::ivec2 left_top{};
	glm::ivec2 advance{};
	std::span<std::byte const> alpha_channels{};
	GlyphIndex glyph_index{};

	[[nodiscard]] constexpr auto operator[](int const x, int const y) const -> std::byte {
		auto const index = std::size_t((y * size.x) + x);
		if (index >= alpha_channels.size()) { return {}; }
		return alpha_channels[index];
	}
};

struct Glyph {
	Codepoint codepoint{};
	glm::vec2 size{};
	glm::vec2 left_top{};
	glm::vec2 advance{};
	UvRect uv_rect{};
	GlyphIndex index{};

	[[nodiscard]] constexpr auto rect(glm::vec2 const baseline, float const scale = 1.0f) const -> Rect<> {
		return {.lt = baseline + scale * left_top, .rb = baseline + scale * (left_top + glm::vec2{size.x, -size.y})};
	}

	[[nodiscard]] constexpr auto is_empty() const -> bool { return advance == glm::vec2{0.0f} && size == glm::vec2{0.0f}; }
};

struct Atlas {
	ColorBitmap bitmap{};
	std::vector<Glyph> glyphs{};
	std::uint32_t height{};
};

struct GlyphLayout {
	gsl::not_null<Glyph const*> glyph;
	glm::vec2 baseline{};
};

struct TextInput {
	std::string_view text;
	std::span<Glyph const> glyphs;
	std::uint32_t height;
	float n_line_height{1.5f};
};

class Typeface {
  public:
	static constexpr auto padding_v = glm::ivec2{2};

	[[nodiscard]] static auto default_codepoints() -> std::span<Codepoint const>;

	Typeface() = default;

	explicit Typeface(std::vector<std::byte> font) { load(std::move(font)); }

	auto load(std::vector<std::byte> font) -> bool;
	[[nodiscard]] auto is_loaded() const -> bool;

	[[nodiscard]] auto get_name() const -> klib::CString;

	auto load_slot(Slot& out, std::uint32_t height, Codepoint codepoint) -> bool;

	[[nodiscard]] auto has_kerning() const -> bool;
	[[nodiscard]] auto get_kerning(std::uint32_t height, GlyphIndex left, GlyphIndex right) const -> glm::ivec2;

	[[nodiscard]] auto build_atlas(std::uint32_t height, std::span<Codepoint const> codepoints = default_codepoints(), glm::ivec2 padding = padding_v) -> Atlas;

	/// \brief Build GlyphLayouts for given TextInput.
	/// \returns Position of cursor (for the next glyph).
	auto push_layouts(std::vector<GlyphLayout>& out, TextInput const& input, bool use_tofu = true) const -> glm::vec2;

	explicit operator bool() const { return is_loaded(); }

  private:
	struct Impl;
	struct Deleter {
		void operator()(Impl* ptr) const noexcept;
	};
	std::unique_ptr<Impl, Deleter> m_impl{};
};

[[nodiscard]] auto glyph_or_fallback(std::span<Glyph const> glyphs, Codepoint codepoint, bool use_tofu = true) -> Glyph const&;
[[nodiscard]] auto glyph_bounds(std::span<GlyphLayout const> glyph_layouts) -> Rect<>;
} // namespace kvf::ttf
