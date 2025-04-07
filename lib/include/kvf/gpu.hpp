#pragma once
#include <vulkan/vulkan.hpp>

namespace kvf {
struct Gpu {
	vk::PhysicalDevice device{};
	vk::PhysicalDeviceProperties properties{};
	vk::PhysicalDeviceFeatures features{};
};
} // namespace kvf
