#pragma once
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <klib/c_string.hpp>
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

[[nodiscard]] inline auto create_window(glm::ivec2 const size, klib::CString const title) -> UniqueWindow {
	if (glfwInit() != GLFW_TRUE) { throw Error{"Failed to initialize GLFW"}; }
	if (glfwVulkanSupported() != GLFW_TRUE) { throw Error{"Vulkan not supported"}; }
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	auto* window = glfwCreateWindow(size.x, size.y, title.c_str(), nullptr, nullptr);
	if (window == nullptr) { throw Error{"Failed to create GLFW Window"}; }
	return UniqueWindow{window};
}
} // namespace kvf
