#pragma once
#include <glm/vec2.hpp>
#include <cstddef>
#include <span>

namespace kvf {
struct Bitmap {
	static constexpr std::uint32_t channels_v{4};

	std::span<std::byte const> bytes{};
	glm::ivec2 size{};
};
} // namespace kvf
