#pragma once
#include <array>

namespace kvf {
constexpr std::size_t buffering_v{KVF_RESOURCE_BUFFERING};
static_assert(buffering_v >= 2 && buffering_v <= 8);

enum struct FrameIndex : std::size_t {};

template <typename Type>
using Buffered = std::array<Type, buffering_v>;
} // namespace kvf
