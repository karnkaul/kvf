#pragma once
#include "kvf/bitmap.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kvf {
enum class Encoding : std::int8_t { Png, Jpg, Tga };

class ImageWriter {
  public:
	auto write_to(std::vector<std::byte>& out, Encoding encoding) const -> bool;

	[[nodiscard]] auto write(Encoding encoding) const -> std::vector<std::byte> {
		auto ret = std::vector<std::byte>{};
		write_to(ret, encoding);
		return ret;
	}

	Bitmap bitmap{};
	int jpg_quality{80};
};
} // namespace kvf
