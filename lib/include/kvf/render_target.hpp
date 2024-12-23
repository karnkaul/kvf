#pragma once
#include <vulkan/vulkan.hpp>

namespace kvf {
struct RenderTarget {
	vk::Image image{};
	vk::ImageView view{};
	vk::Extent2D extent{};
};
} // namespace kvf
