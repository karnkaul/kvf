#pragma once
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <klib/c_string.hpp>
#include <memory>

namespace kvf {
struct WindowDeleter {
	void operator()(GLFWwindow* ptr) const noexcept;
};

using UniqueWindow = std::unique_ptr<GLFWwindow, WindowDeleter>;

[[nodiscard]] auto create_window(glm::ivec2 size, klib::CString title, bool decorated = true) noexcept(false) -> UniqueWindow;
[[nodiscard]] auto create_fullscreen_window(klib::CString title, GLFWmonitor* target = nullptr) noexcept(false) -> UniqueWindow;
} // namespace kvf
