#pragma once
#include <cstddef>

namespace kvf {
inline constexpr std::size_t resource_buffering_v{KVF_RESOURCE_BUFFERING};
static_assert(resource_buffering_v >= 2 && resource_buffering_v <= 8);

inline constexpr bool use_freetype_v{KVF_USE_FREETYPE == 1};
} // namespace kvf
