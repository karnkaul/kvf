#pragma once
#include <kvf/is_positive.hpp>

namespace kvf {
enum class ResizeAspect : std::int8_t { None, FixHeight, FixWidth };

constexpr auto aspect_resize(glm::vec2 size, glm::vec2 const reference, ResizeAspect const aspect) -> glm::vec2 {
	if (!is_positive(size) || !is_positive(reference)) { return {}; }
	switch (aspect) {
	default:
	case ResizeAspect::None: break;
	case ResizeAspect::FixHeight: size.x = size.y * reference.x / reference.y; break;
	case ResizeAspect::FixWidth: size.y = size.x * reference.y / reference.x; break;
	}
	return size;
}
} // namespace kvf
