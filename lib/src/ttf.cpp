#include <klib/unique.hpp>
#include <kvf/constants.hpp>
#include <kvf/ttf.hpp>
#include <log.hpp>
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
} // namespace

struct Typeface::Impl {
	struct Deleter {
		void operator()(FT_Face face) const noexcept { FT_Done_Face(face); }
	};

	std::shared_ptr<Lib> lib{g_data.get_or_make_lib()};
	std::vector<std::byte> font{};
	klib::Unique<FT_Face, Deleter> face{};
};

void Typeface::Deleter::operator()(Impl* ptr) const noexcept { std::default_delete<Impl>{}(ptr); }

Typeface::Typeface() : m_impl(new Impl) {}

auto Typeface::load(std::vector<std::byte> font) -> bool {
	if (!m_impl) { m_impl.reset(new Impl); } // NOLINT(cppcoreguidelines-owning-memory)
	if (!m_impl->lib) { return false; }

	m_impl->face = g_data.load_face(m_impl->lib->lib.get(), font.data(), font.size());
	if (m_impl->face.is_identity()) { return false; }

	m_impl->font = std::move(font);
	return true;
}

auto Typeface::set_height(std::uint32_t const height) -> bool {
	if (!is_loaded()) { return false; }
	return FT_Set_Pixel_Sizes(static_cast<FT_Face>(m_impl->face.get()), 0, FT_UInt(height)) == FT_Err_Ok;
}

auto Typeface::is_loaded() const -> bool { return m_impl && !m_impl->face.is_identity(); }

auto Typeface::load_slot(Slot& out, Codepoint const codepoint) -> bool {
	if (!is_loaded()) { return false; }
	if (FT_Load_Char(m_impl->face.get(), FT_ULong(codepoint), FT_LOAD_RENDER) != FT_Err_Ok) { return false; }

	auto const* glyph = m_impl->face.get()->glyph;
	if (glyph == nullptr) { return false; }

	auto const* ptr = static_cast<void const*>(glyph->bitmap.buffer);
	auto const size = glyph->bitmap.width * glyph->bitmap.rows;
	out = Slot{
		.size = {glyph->bitmap.width, glyph->bitmap.rows},
		.left_top = {glyph->bitmap_left, glyph->bitmap_top},
		.advance = {glyph->advance.x, glyph->advance.y},
		.alpha_channels = std::span{static_cast<std::byte const*>(ptr), std::size_t(size)},
	};
	return true;
}

#else

Typeface::Typeface() = default;

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::load(std::vector<std::byte> /*font*/) -> bool { return false; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::set_height(std::uint32_t const /*height*/) -> bool { return false; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::is_loaded() const -> bool { return false; }

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto Typeface::load_slot(Slot& /*out*/, Codepoint const /*codepoint*/) -> bool { return false; }

#endif
} // namespace kvf::ttf
