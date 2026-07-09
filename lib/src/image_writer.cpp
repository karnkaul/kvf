#include "kvf/image_writer.hpp"
#include <stb/stb_image_write.h>

namespace kvf {
namespace {
class Writer {
  public:
	explicit Writer(std::vector<std::byte>& out) : m_out(out) {}

	auto encode_png(Bitmap const& bitmap) -> bool {
		auto const stride = bitmap.size.x * comp_v;
		return stbi_write_png_to_func(&write_callback, this, bitmap.size.x, bitmap.size.y, comp_v, bitmap.bytes.data(), stride);
	}

	auto encode_jpg(Bitmap const& bitmap, int const quality) -> bool {
		return stbi_write_jpg_to_func(&write_callback, this, bitmap.size.x, bitmap.size.y, comp_v, bitmap.bytes.data(), quality) != 0;
	}

	auto encode_tga(Bitmap const& bitmap) -> bool {
		return stbi_write_tga_to_func(&write_callback, this, bitmap.size.x, bitmap.size.y, comp_v, bitmap.bytes.data());
	}

  private:
	static constexpr auto comp_v = int(Bitmap::channels_v);

	void on_write(void const* data, int const size) const {
		auto const in = std::span{static_cast<std::byte const*>(data), std::size_t(size)};
		m_out.append_range(in);
	}

	static void write_callback(void* context, void* data, int size) { static_cast<Writer*>(context)->on_write(data, size); }

	std::vector<std::byte>& m_out;
};
} // namespace

auto ImageWriter::write_to(std::vector<std::byte>& out, Encoding encoding) const -> bool {
	auto writer = Writer{out};
	switch (encoding) {
	case Encoding::Png: return writer.encode_png(bitmap);
	case Encoding::Jpg: return writer.encode_jpg(bitmap, jpg_quality);
	case Encoding::Tga: return writer.encode_tga(bitmap);
	default: return false;
	}
}
} // namespace kvf
