#include <imgui.h>
#include <klib/log.hpp>
#include <kvf/render_device.hpp>
#include <kvf/render_pass.hpp>
#include <kvf/util.hpp>
#include <kvf/window.hpp>
#include <chrono>
#include <print>

namespace {
namespace ch = std::chrono;

struct App {
	explicit App() : m_window(make_window()), m_device(m_window.get()), m_color_pass(&m_device, m_device.get_framebuffer_extent()) {}

	void run() {
		m_color_pass.set_color_target();
		m_color_pass.set_depth_target();
		m_color_pass.clear_color = vk::ClearColorValue{std::array{1.0f, 0.0f, 0.0f, 1.0f}};

		while (glfwWindowShouldClose(m_window.get()) != GLFW_TRUE) {
			auto command_buffer = m_device.next_frame();

			ImGui::ShowDemoWindow();

			m_color_pass.resize(m_device.get_framebuffer_extent());
			m_color_pass.begin_render(command_buffer);
			m_color_pass.end_render();

			m_device.render(m_color_pass.render_target());
		}
	}

  private:
	auto make_window() -> kvf::UniqueWindow {
		auto ret = kvf::create_window("kvf example", 1280, 720);
		glfwSetWindowUserPointer(ret.get(), this);
		glfwSetKeyCallback(ret.get(), [](GLFWwindow* w, int const key, int const /*scancode*/, int action, int const mods) {
			if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE && mods == 0) { glfwSetWindowShouldClose(w, GLFW_TRUE); }
			if (key == GLFW_KEY_I && action == GLFW_RELEASE && mods == 0) {
				auto& self = *static_cast<App*>(glfwGetWindowUserPointer(w));
				self.m_device.set_render_imgui(!self.m_device.get_render_imgui());
			}
		});
		return ret;
	}

	kvf::UniqueWindow m_window{};
	kvf::RenderDevice m_device;

	kvf::RenderPass m_color_pass;
};
} // namespace

auto main() -> int {
	auto const log_file = klib::log::File{"kvf-example.log"};
	try {
		App{}.run();
	} catch (std::exception const& e) {
		klib::log::error("PANIC", "{}", e.what());
		return EXIT_FAILURE;
	} catch (...) {
		klib::log::error("PANIC", "Unknown");
		return EXIT_FAILURE;
	}
}
