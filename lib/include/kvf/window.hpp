#pragma once
#include <GLFW/glfw3.h>
#include <kvf/error.hpp>
#include <vulkan/vulkan.hpp>
#include <memory>

namespace kvf {
struct WindowDeleter {
	void operator()(GLFWwindow* ptr) const noexcept {
		glfwDestroyWindow(ptr);
		glfwTerminate();
	}
};

using UniqueWindow = std::unique_ptr<GLFWwindow, WindowDeleter>;

[[nodiscard]] inline auto create_window(char const* title, int const width, int const height) -> UniqueWindow {
	if (glfwInit() != GLFW_TRUE) { throw Error{"Failed to initialize GLFW"}; }
	if (glfwVulkanSupported() != GLFW_TRUE) { throw Error{"Vulkan not supported"}; }
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	auto* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (window == nullptr) { throw Error{"Failed to create GLFW Window"}; }
	return UniqueWindow{window};
}
} // namespace kvf
