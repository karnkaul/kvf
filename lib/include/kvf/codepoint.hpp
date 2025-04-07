#pragma once
#include <cstdint>

namespace kvf {
// NOLINTNEXTLINE(performance-enum-size)
enum struct Codepoint : std::uint32_t {
	Tofu = 0,
	Space = 32,
	AsciiFirst = Space,
	AsciiLast = 126,
};
} // namespace kvf
