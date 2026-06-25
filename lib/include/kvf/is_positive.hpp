#pragma once
#include "klib/concepts.hpp"
#include <glm/vec2.hpp>

namespace kvf {
template <klib::NumberT Type>
constexpr auto is_positive(Type const t) -> bool {
	return t > Type(0);
}

template <klib::NumberT Type>
constexpr auto is_positive(glm::tvec2<Type> const t) -> bool {
	return is_positive(t.x) && is_positive(t.y);
}
} // namespace kvf
