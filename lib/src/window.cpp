#include "kvf/window.hpp"
#include "klib/log/log.hpp"
#include "kvf/panic.hpp"

namespace {
void glfw_init() {
	static auto const on_error = [](int const code, char const* description) {
		static constexpr std::string_view tag_v{"glfw"};
		klib::log::error(tag_v, "{} ({})", description, code);
	};
	glfwSetErrorCallback(on_error);
	if (glfwInit() != GLFW_TRUE) { throw kvf::Panic{"Failed to initialize GLFW"}; }
	if (glfwVulkanSupported() != GLFW_TRUE) { throw kvf::Panic{"Vulkan not supported"}; }
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
}
} // namespace

void kvf::WindowDeleter::operator()(GLFWwindow* ptr) const noexcept {
	glfwDestroyWindow(ptr);
	glfwTerminate();
}

auto kvf::create_window(glm::ivec2 const size, klib::CString const title, std::span<WindowHint const> hints) -> UniqueWindow {
	glfw_init();
	for (auto const& hint : hints) { glfwWindowHint(hint.hint, hint.value); }
	auto* window = glfwCreateWindow(size.x, size.y, title.c_str(), nullptr, nullptr);
	if (window == nullptr) { throw Panic{"Failed to create GLFW Window"}; }
	glfwSetWindowSize(window, size.x, size.y);
	return UniqueWindow{window};
}

auto kvf::create_fullscreen_window(klib::CString const title) -> UniqueWindow {
	glfw_init();
	GLFWmonitor* target = glfwGetPrimaryMonitor();
	GLFWvidmode const* mode = glfwGetVideoMode(target);
	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
	glfwWindowHint(GLFW_CENTER_CURSOR, GLFW_TRUE);
	auto* window = glfwCreateWindow(mode->width, mode->height, title.c_str(), target, nullptr);
	if (window == nullptr) { throw Panic{"Failed to create GLFW Window"}; }
	return UniqueWindow{window};
}
