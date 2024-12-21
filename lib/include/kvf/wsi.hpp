#pragma once
#include <klib/polymorphic.hpp>
#include <vulkan/vulkan.hpp>
#include <span>

namespace kvf {
struct Gpu {
	vk::PhysicalDevice device{};
	vk::PhysicalDeviceProperties properties{};
	vk::PhysicalDeviceFeatures features{};
};

class IWsi : public klib::Polymorphic {
  public:
	using Gpu = kvf::Gpu;

	[[nodiscard]] virtual auto get_instance_extensions() const -> std::span<char const* const> = 0;
	[[nodiscard]] virtual auto compare_gpus(Gpu const& a, Gpu const& b) const -> bool = 0;
	[[nodiscard]] virtual auto create_surface(vk::Instance instance) const -> vk::SurfaceKHR = 0;
	[[nodiscard]] virtual auto get_framebuffer_size() const -> vk::Extent2D = 0;
};
} // namespace kvf
