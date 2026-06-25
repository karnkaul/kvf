#pragma once
#include "klib/concepts.hpp"
#include <bit>
#include <cstddef>
#include <span>

namespace kvf {
template <std::size_t Size>
struct ByteArray {
	constexpr operator std::span<std::byte>() { return bytes; }
	constexpr operator std::span<std::byte const>() const { return bytes; }

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
	std::byte bytes[Size]{};
};

template <klib::MemcpyAble Type>
constexpr auto to_byte_array(Type const& t) {
	return std::bit_cast<ByteArray<sizeof(t)>>(t);
}
} // namespace kvf
