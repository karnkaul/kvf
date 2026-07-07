#pragma once
#include "klib/base_types.hpp"
#include <vulkan/vulkan.hpp>
#include <gsl/pointers>

namespace kvf {
struct Gpu {
	class Selector;

	vk::PhysicalDevice device{};
	vk::PhysicalDeviceProperties properties{};
	vk::PhysicalDeviceFeatures features{};
};

class Gpu::Selector : public klib::Polymorphic {
  public:
	[[nodiscard]] virtual auto select(std::span<Gpu const> gpus) const -> gsl::not_null<Gpu const*> {
		for (auto const& gpu : gpus) {
			if (gpu.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) { return &gpu; }
		}
		return &gpus.front();
	}
};
} // namespace kvf
