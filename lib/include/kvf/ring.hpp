#pragma once
#include "kvf/constants.hpp"
#include <array>

namespace kvf {
template <typename Type>
using Ring = std::array<Type, resource_buffering_v>;
} // namespace kvf
