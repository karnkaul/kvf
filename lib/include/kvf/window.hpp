#pragma once
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <klib/c_string.hpp>
#include <memory>
#include <span>

namespace kvf {
struct WindowDeleter {
	void operator()(GLFWwindow* ptr) const noexcept;
};

struct WindowHint {
	int hint{};
	int value{};
};

using UniqueWindow = std::unique_ptr<GLFWwindow, WindowDeleter>;

[[nodiscard]] auto create_window(glm::ivec2 size, klib::CString title, std::span<WindowHint const> hints = {}) noexcept(false) -> UniqueWindow;
[[nodiscard]] auto create_fullscreen_window(klib::CString title) noexcept(false) -> UniqueWindow;
} // namespace kvf
