#pragma once
#include <kvf/color_bitmap.hpp>
#include <kvf/rect.hpp>
#include <cstddef>
#include <cstdint>
#include <gsl/pointers>
#include <memory>
#include <optional>
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
};

class Typeface {
  public:
	[[nodiscard]] static auto default_codepoints() -> std::span<Codepoint const>;

	Typeface();

	explicit Typeface(std::vector<std::byte> font) { load(std::move(font)); }

	auto load(std::vector<std::byte> font) -> bool;
	[[nodiscard]] auto is_loaded() const -> bool;

	auto set_height(std::uint32_t height) -> bool;
	auto load_slot(Slot& out, Codepoint codepoint) -> bool;

	[[nodiscard]] auto has_kerning() const -> bool;
	[[nodiscard]] auto get_kerning(GlyphIndex left, GlyphIndex right) const -> glm::ivec2;

	[[nodiscard]] auto build_atlas(std::span<Codepoint const> codepoints = default_codepoints(), glm::ivec2 padding = glm::ivec2{2}) -> Atlas;

	explicit operator bool() const { return is_loaded(); }

  private:
	struct Impl;
	struct Deleter {
		void operator()(Impl* ptr) const noexcept;
	};
	std::unique_ptr<Impl, Deleter> m_impl{};
};

struct IterationEntry {
	gsl::not_null<Glyph const*> glyph;
	char ch{};
	glm::vec2 kerning{};
};

struct GlyphIterator {
	using Entry = IterationEntry;

	[[nodiscard]] static auto advance(glm::vec2 position, Entry const& entry) -> glm::vec2 { return position + entry.glyph->advance + entry.kerning; }

	[[nodiscard]] auto glyph_or_fallback(Codepoint codepoint) const -> Glyph const&;

	[[nodiscard]] auto line_bounds(std::string_view line) const -> Rect<>;
	[[nodiscard]] auto next_glyph_position(std::string_view line) const -> glm::vec2;

	template <typename F>
		requires(std::invocable<F, Entry const&>)
	void iterate(std::string_view const text, F func) const {
		auto previous = std::optional<GlyphIndex>{};
		for (char const c : text) {
			auto const codepoint = Codepoint(c);
			auto const& glyph = glyph_or_fallback(codepoint);
			auto entry = Entry{.glyph = &glyph, .ch = c};
			if (face != nullptr && previous) { entry.kerning = face->get_kerning(*previous, glyph.index); }
			previous = glyph.index;
			func(entry);
		}
	}

	Typeface const* face{};
	std::span<Glyph const> glyphs{};
	bool use_tofu{true};
};
} // namespace kvf::ttf
