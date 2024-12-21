#include <kvf/render_device.hpp>
#include <window.hpp>
#include <chrono>
#include <print>

namespace {
namespace ch = std::chrono;

struct App {
	explicit App() : m_window(Window::create("kvf example", 1280, 720)), m_device(&m_window) {}

	void run() {
		glfwSetKeyCallback(m_window.get(), [](GLFWwindow* w, int const key, int const action, int /*scancode*/, int const mods) {
			if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE && mods == 0) { glfwSetWindowShouldClose(w, GLFW_TRUE); }
		});

		m_device.request_present_mode(vk::PresentModeKHR::eFifo);

		while (glfwWindowShouldClose(m_window.get()) != GLFW_TRUE) {
			glfwPollEvents();

			m_device.acquire_next_image();
			m_device.temp_render();
			m_device.present_acquired_image();
		}
	}

  private:
	Window m_window;
	kvf::RenderDevice m_device;
};
} // namespace

auto main() -> int {
	try {
		App{}.run();
	} catch (std::exception const& e) {
		std::println(stderr, "PANIC: {}", e.what());
		return EXIT_FAILURE;
	} catch (...) {
		std::println(stderr, "PANIC!");
		return EXIT_FAILURE;
	}
}
