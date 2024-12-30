#pragma once
#include <kvf/constants.hpp>
#include <array>

namespace kvf {
enum struct FrameIndex : std::size_t {};

template <typename Type>
using Buffered = std::array<Type, resource_buffering_v>;
} // namespace kvf
