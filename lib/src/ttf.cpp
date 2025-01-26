#include <klib/unique.hpp>
#include <kvf/constants.hpp>
#include <kvf/ttf.hpp>
#include <log.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

#if KVF_USE_FREETYPE == 1
#include <ft2build.h>

#include FT_FREETYPE_H
#endif

namespace kvf::ttf {
#if KVF_USE_FREETYPE == 1

namespace {
struct Lib {
	struct Deleter {
		void operator()(FT_Library lib) const noexcept { FT_Done_FreeType(lib); }
	};

	operator FT_Library() const { return lib.get(); }

	klib::Unique<FT_Library, Deleter> lib{};
};

struct {
	auto get_or_make_lib() -> std::shared_ptr<Lib> {
		auto lock = std::scoped_lock{mutex};
		if (auto ret = lib.lock()) { return ret; }

		auto* p_lib = FT_Library{};
		if (FT_Init_FreeType(&p_lib) != FT_Err_Ok) { return {}; }
		auto ret = std::make_shared<Lib>(Lib{.lib = p_lib});
		lib = ret;
		return ret;
	}

	auto load_face(FT_Library lib, void const* data, std::size_t const size) -> FT_Face {
		auto* p_face = FT_Face{};
		auto lock = std::scoped_lock{mutex};
		if (FT_New_Memory_Face(lib, static_cast<FT_Byte const*>(data), FT_Long(size), 0, &p_face) != FT_Err_Ok) { return {}; }
		return p_face;
	}

	std::weak_ptr<Lib> lib{};
	std::mutex mutex{};
} g_data{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

constexpr auto pot(int const in) {
	auto ret = 1;
	while (ret < in && ret < std::numeric_limits<int>::max()) { ret <<= 1; }
	return ret;
}

struct BuildAtlas {
	static constexpr int max_size_v{8 * 1024};

	auto operator()(Typeface& face, std::uint32_t const height, std::span<Codepoint const> codepoints, glm::ivec2 const pad) -> Atlas {
		if (!face.is_loaded()) { return {}; }

		m_pad = pad;
		load_entries(face, height, codepoints);
		store_pixels();
		if (m_atlas_size.x > max_size_v || m_atlas_size.y > max_size_v) { return {}; }

		return finalize(height);
	}

  private:
	struct Alpha {
		std::size_t first{};
		std::size_t count{};
	};

	struct Entry {
		Codepoint codepoint{};
		Slot slot{};
		Alpha alpha{};
		Rect<int> uv_rect{};
	};

	struct Pixel {
		std::byte alpha{};
		glm::ivec2 coords{};
	};

	void load_entry(Typeface& face, std::uint32_t const height, Codepoint const codepoint) {
		auto slot = Slot{};
		auto alpha = Alpha{};
		if (face.load_slot(slot, height, codepoint)) {
			alpha = Alpha{.first = m_alphas.size(), .count = slot.alpha_channels.size()};
			if (alpha.count > 0) {
				m_alphas.resize(m_alphas.size() + alpha.count);
				auto const dst = std::span{m_alphas}.subspan(alpha.first);
				std::memcpy(dst.data(), slot.alpha_channels.data(), dst.size());
			}
			m_max_glyph_width = std::max(m_max_glyph_width, slot.size.x);
		}
		m_entries.push_back(Entry{.codepoint = codepoint, .slot = slot, .alpha = alpha});
	}

	void load_entries(Typeface& face, std::uint32_t const height, std::span<Codepoint const> codepoints) {
		m_entries.reserve(codepoints.size());
		for (auto const codepoint : codepoints) { load_entry(face, height, codepoint); }

		auto const fcolumns = std::sqrt(m_entries.size());
		auto const columns = int(std::ceil(fcolumns));
		m_atlas_size.x = pot(((m_max_glyph_width + m_pad.x) * columns) + m_pad.x);
	}

	void store_pixels() {
		m_cursor = m_pad;
		for (auto& entry : m_entries) {
			if (entry.alpha.count == 0) { continue; }
			entry.slot.alpha_channels = std::span{m_alphas}.subspan(entry.alpha.first, entry.alpha.count);
			auto const width = m_cursor.x + entry.slot.size.x + m_pad.x;
			if (width > m_atlas_size.x) { next_line(); }
			entry.uv_rect.lt = m_cursor;
			entry.uv_rect.rb = m_cursor + entry.slot.size;
			m_line_height = std::max(m_line_height, entry.slot.size.y);
			for (int y = 0; y < entry.slot.size.y; ++y) {
				for (int x = 0; x < entry.slot.size.x; ++x) { m_pixels.push_back(Pixel{.alpha = entry.slot[x, y], .coords = m_cursor + glm::ivec2{x, y}}); }
			}
			m_cursor.x += entry.slot.size.x + m_pad.x;
		}
		m_atlas_size.y = m_cursor.y + m_line_height + m_pad.y;
	}

	void next_line() {
		m_cursor.x = m_pad.x;
		m_cursor.y += m_line_height + m_pad.y;
		m_line_height = 0;
	}

	auto finalize(std::uint32_t const height) -> Atlas {
		auto ret = Atlas{};

		m_atlas_size.y = pot(m_atlas_size.y);
		ret.bitmap = ColorBitmap{m_atlas_size};
		ret.height = height;

		ret.glyphs.reserve(m_entries.size());
		auto const fsize = glm::vec2(m_atlas_size);
		for (auto const& entry : m_entries) {
			auto const uv_rect = Rect<>{
				.lt = glm::vec2(entry.uv_rect.lt) / fsize,
				.rb = glm::vec2(entry.uv_rect.rb) / fsize,
			};
			ret.glyphs.push_back(Glyph{
				.codepoint = entry.codepoint,
				.size = entry.slot.size,
				.left_top = entry.slot.left_top,
				.advance = entry.slot.advance,
				.uv_rect = uv_rect,
				.index = entry.slot.glyph_index,
			});
		}

		for (auto const& pixel : m_pixels) { ret.bitmap[pixel.coords.x, pixel.coords.y] = GlmColor{0xff, 0xff, 0xff, std::uint8_t(pixel.alpha)}; }

		return ret;
	}

	glm::ivec2 m_pad{};

	std::vector<std::byte> m_alphas{};
	std::vector<Entry> m_entries{};
	int m_max_glyph_width{};

	std::vector<Pixel> m_pixels{};
	glm::ivec2 m_atlas_size{};
	glm::ivec2 m_cursor{};
	int m_line_height{};
};
} // namespace

struct Typeface::Impl {
	struct Deleter {
		void operator()(FT_Face face) const noexcept { FT_Done_Face(face); }
	};

	std::shared_ptr<Lib> lib{g_data.get_or_make_lib()};
	std::vector<std::byte> font{};
	klib::Unique<FT_Face, Deleter> face{};
	bool has_kerning{};
};

void Typeface::Deleter::operator()(Impl* ptr) const noexcept { std::default_delete<Impl>{}(ptr); }

auto Typeface::default_codepoints() -> std::span<Codepoint const> {
	static auto const ret = [&] {
		static constexpr auto count_v = std::size_t(Codepoint::AsciiLast) - std::size_t(Codepoint::AsciiFirst) + 2;
		auto ret = std::array<Codepoint, count_v>{};
		auto index = std::size_t{};
		ret.at(index++) = Codepoint::Tofu;
		for (auto c = Codepoint::AsciiFirst; c <= Codepoint::AsciiLast; c = Codepoint(int(c) + 1)) { ret.at(index++) = c; }
		return ret;
	}();
	return ret;
}

auto Typeface::load(std::vector<std::byte> font) -> bool {
	if (!m_impl) { m_impl.reset(new Impl); } // NOLINT(cppcoreguidelines-owning-memory)
	if (!m_impl->lib) { return false; }

	FT_Face face = g_data.load_face(m_impl->lib->lib.get(), font.data(), font.size());
	if (face == nullptr) { return false; }

	m_impl->font = std::move(font);
	m_impl->face = face;
	m_impl->has_kerning = FT_HAS_KERNING(face);

	return true;
}

auto Typeface::is_loaded() const -> bool { return m_impl && !m_impl->face.is_identity(); }

auto Typeface::get_name() const -> klib::CString {
	if (!m_impl) { return {}; }
	return klib::CString{FT_Get_Postscript_Name(m_impl->face.get())};
}

auto Typeface::load_slot(Slot& out, std::uint32_t const height, Codepoint const codepoint) -> bool {
	if (!is_loaded()) { return false; }
	if (FT_Set_Pixel_Sizes(m_impl->face.get(), 0, FT_UInt(height)) != FT_Err_Ok) { return false; }
	if (FT_Load_Char(m_impl->face.get(), FT_ULong(codepoint), FT_LOAD_RENDER) != FT_Err_Ok) { return false; }

	auto const* glyph = m_impl->face.get()->glyph;
	if (glyph == nullptr) { return false; }

	auto const* ptr = static_cast<void const*>(glyph->bitmap.buffer);
	auto const size = glyph->bitmap.width * glyph->bitmap.rows;
	out = Slot{
		.size = {glyph->bitmap.width, glyph->bitmap.rows},
		.left_top = {glyph->bitmap_left, glyph->bitmap_top},
		.advance = {glyph->advance.x >> 6, glyph->advance.y >> 6},
		.alpha_channels = std::span{static_cast<std::byte const*>(ptr), std::size_t(size)},
		.glyph_index = GlyphIndex{FT_Get_Char_Index(m_impl->face.get(), FT_ULong(codepoint))},
	};
	return true;
}

auto Typeface::has_kerning() const -> bool {
	if (!is_loaded()) { return false; }
	return m_impl->has_kerning;
}

auto Typeface::get_kerning(std::uint32_t const height, GlyphIndex const left, GlyphIndex const right) const -> glm::ivec2 {
	if (!has_kerning()) { return {}; }
	if (FT_Set_Pixel_Sizes(m_impl->face.get(), 0, FT_UInt(height)) != FT_Err_Ok) { return {}; }
	auto delta = FT_Vector{};
	FT_Get_Kerning(m_impl->face.get(), FT_UInt(left), FT_UInt(right), FT_KERNING_DEFAULT, &delta);
	return {delta.x >> 6, delta.y >> 6};
}

auto Typeface::build_atlas(std::uint32_t const height, std::span<Codepoint const> codepoints, glm::ivec2 const glyph_padding) -> Atlas {
	if (FT_Set_Pixel_Sizes(m_impl->face.get(), 0, FT_UInt(height)) != FT_Err_Ok) { return {}; }
	return BuildAtlas{}(*this, height, codepoints, glyph_padding);
}

#else

// NOLINTNEXTLINE(readability-convert-member-functions-to-static, performance-unnecessary-value-param)
auto Typeface::load(std::vector<std::byte> /*font*/) -> bool { return false; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::is_loaded() const -> bool { return false; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::get_name() const -> klib::CString { return {}; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::load_slot(Slot& /*out*/, std::uint32_t /*height*/, Codepoint /*codepoint*/) -> bool { return false; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::has_kerning() const -> bool { return false; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::get_kerning(std::uint32_t /*height*/, GlyphIndex /*left*/, GlyphIndex /*right*/) const -> glm::ivec2 { return {}; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::build_atlas(std::uint32_t /*height*/, std::span<Codepoint const> /*codepoints*/, glm::ivec2 /*glyph_padding*/) -> Atlas { return {}; }

#endif

auto Typeface::push_layouts(std::vector<GlyphLayout>& out, TextInput const& input, bool use_tofu) const -> glm::vec2 {
	if (!is_loaded() || input.text.empty() || input.glyphs.empty()) { return {}; }
	out.reserve(out.size() + input.text.size());
	auto baseline = glm::vec2{};
	Glyph const* previous = nullptr;
	for (char const c : input.text) {
		if (c == '\n') {
			baseline.x = 0.0f;
			baseline.y -= input.n_line_height * float(input.height);
			continue;
		}
		auto const codepoint = Codepoint(c);
		auto const& glyph = glyph_or_fallback(input.glyphs, codepoint, use_tofu);
		if (previous != nullptr) { baseline += get_kerning(input.height, previous->index, glyph.index); }
		auto const glyph_layout = GlyphLayout{.glyph = &glyph, .baseline = baseline};
		out.push_back(glyph_layout);
		baseline += glyph.advance;
		previous = &glyph;
	}
	return baseline;
}
} // namespace kvf::ttf

auto kvf::ttf::glyph_or_fallback(std::span<Glyph const> glyphs, Codepoint const codepoint, bool const use_tofu) -> Glyph const& {
	auto it = std::ranges::find_if(glyphs, [codepoint](Glyph const& g) { return g.codepoint == codepoint; });
	if (it == glyphs.end() && use_tofu) {
		it = std::ranges::find_if(glyphs, [](Glyph const& g) { return g.codepoint == Codepoint::Tofu; });
	}
	if (it == glyphs.end()) {
		static constexpr auto null_v = Glyph{};
		return null_v;
	}
	return *it;
}

auto kvf::ttf::glyph_bounds(std::span<GlyphLayout const> glyph_layouts) -> Rect<> {
	auto ret = Rect<>{};
	if (glyph_layouts.empty()) { return ret; }
	ret.lt.x = std::numeric_limits<float>::max();
	for (auto const& layout : glyph_layouts) {
		auto const rect = layout.glyph->rect(layout.baseline);
		ret.lt.x = std::min(ret.lt.x, rect.lt.x);
		ret.lt.y = std::max(ret.lt.y, rect.lt.y);
		ret.rb.x = std::max(ret.rb.x, rect.rb.x);
		ret.rb.y = std::min(ret.rb.y, rect.rb.y);
	}
	return ret;
}
