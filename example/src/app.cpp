#include <imgui.h>
#include <app.hpp>
#include <klib/fixed_string.hpp>
#include <kvf/error.hpp>
#include <log.hpp>
#include <scenes/image_viewer.hpp>
#include <scenes/standalone.hpp>
#include <scenes/triangle.hpp>

namespace kvf::example {
App::App(std::string_view const build_version) : m_window(make_window(build_version)), m_device(m_window.get()) {
	m_factories.push_back(Factory{.name = "Standalone", .create = [this] { return std::make_unique<Standalone>(&m_device, m_assets_dir); }});
	m_factories.push_back(Factory{.name = "Image Viewer", .create = [this] { return std::make_unique<ImageViewer>(&m_device, m_assets_dir); }});
	m_factories.push_back(Factory{.name = "Triangle", .create = [this] { return std::make_unique<Triangle>(&m_device, m_assets_dir); }});
}

void App::run(std::string_view const assets_dir) {
	m_assets_dir = assets_dir;
	m_current_factory = &m_factories.front();
	m_scene = m_current_factory->create();

	while (!m_device.is_window_closing()) {
		auto command_buffer = m_device.next_frame();
		draw_menu();
		m_scene->update(command_buffer);
		draw_error_modal();
		m_device.render(m_scene->get_render_target(), m_scene->get_render_filter());
	}
}

auto App::make_window(std::string_view const build_version) -> kvf::UniqueWindow {
	auto const title = klib::FixedString{"kvf example [{}]", build_version};
	auto ret = kvf::create_window(title.c_str(), 800, 600);
	glfwSetWindowUserPointer(ret.get(), this);
	glfwSetKeyCallback(ret.get(), [](GLFWwindow* w, int const key, int const /*scancode*/, int action, int const mods) {
		auto& self = *static_cast<App*>(glfwGetWindowUserPointer(w));
		if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE && mods == 0) { self.m_device.set_window_closing(true); }
		auto const input = Scene::KeyInput{.key = key, .action = action, .mods = mods};
		self.m_scene->on_key(input);
	});
	glfwSetDropCallback(ret.get(), [](GLFWwindow* w, int const count, char const** paths) {
		auto const span = std::span{paths, std::size_t(count)};
		static_cast<App*>(glfwGetWindowUserPointer(w))->m_scene->on_drop(span);
	});
	return ret;
}

void App::draw_menu() {
	if (!ImGui::BeginMainMenuBar()) { return; }
	if (ImGui::BeginMenu("File")) {
		if (ImGui::MenuItem("Exit")) { m_device.set_window_closing(true); }
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Scenes")) {
		auto* new_factory = m_current_factory;
		for (auto& factory : m_factories) {
			if (ImGui::MenuItem(factory.name.c_str(), nullptr, m_current_factory == &factory)) { new_factory = &factory; }
		}

		if (new_factory != m_current_factory) {
			try {
				m_scene = new_factory->create();
				m_current_factory = new_factory;
			} catch (Error const& e) {
				auto const message = std::format("Failed to create scene {}\n{}", new_factory->name.as_view(), e.what());
				m_scene->open_error_modal(message);
			}
		}
		ImGui::EndMenu();
	}
	ImGui::EndMainMenuBar();
}

void App::draw_error_modal() const {
	auto& modal = m_scene->m_modal;

	if (modal.set_open) {
		ImGui::OpenPopup("Error!");
		log::error("{}", modal.message);
		modal.set_open = false;
	}

	if (!ImGui::BeginPopupModal("Error!")) { return; }

	ImGui::TextUnformatted(modal.message.c_str());
	if (ImGui::Button("Close")) {
		ImGui::CloseCurrentPopup();
		modal = {};
	}

	ImGui::EndPopup();
}
} // namespace kvf::example
