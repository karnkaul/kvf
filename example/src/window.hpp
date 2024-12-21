#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <kvf/wsi.hpp>
#include <memory>
#include <stdexcept>

struct WindowDeleter {
	void operator()(GLFWwindow* ptr) const noexcept {
		glfwDestroyWindow(ptr);
		glfwTerminate();
	}
};

struct Window : std::unique_ptr<GLFWwindow, WindowDeleter>, kvf::IWsi {
	using std::unique_ptr<GLFWwindow, WindowDeleter>::unique_ptr;

	[[nodiscard]] static auto create(char const* title, int const width, int const height) {
		if (glfwInit() != GLFW_TRUE) { throw std::runtime_error{"Failed to initialize GLFW"}; }
		if (glfwVulkanSupported() != GLFW_TRUE) { throw std::runtime_error{"Vulkan not supported"}; }
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		auto* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
		if (window == nullptr) { throw std::runtime_error{"Failed to create GLFW Window"}; }
		return Window{window};
	}

	[[nodiscard]] auto get_instance_extensions() const -> std::span<char const* const> final {
		auto count = std::uint32_t{};
		auto const* first = glfwGetRequiredInstanceExtensions(&count);
		return {first, count};
	}

	[[nodiscard]] auto compare_gpus(Gpu const& a, Gpu const& /*b*/) const -> bool final {
		return a.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	}

	[[nodiscard]] auto create_surface(vk::Instance instance) const -> vk::SurfaceKHR final {
		VkSurfaceKHR ret{};
		glfwCreateWindowSurface(instance, get(), nullptr, &ret);
		return ret;
	}

	[[nodiscard]] auto get_framebuffer_size() const -> vk::Extent2D final {
		auto width = int{};
		auto height = int{};
		glfwGetFramebufferSize(get(), &width, &height);
		return {std::uint32_t(width), std::uint32_t(height)};
	}
};
