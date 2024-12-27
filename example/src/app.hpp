#pragma once
#include <klib/c_string.hpp>
#include <kvf/device_block.hpp>
#include <kvf/render_device.hpp>
#include <kvf/window.hpp>
#include <scene.hpp>
#include <functional>
#include <memory>

namespace kvf::example {
class App {
  public:
	explicit App(std::string_view build_version);

	void run(std::string_view assets_dir);

  private:
	struct Factory {
		klib::CString name{};
		std::move_only_function<std::unique_ptr<Scene>()> create{};
	};

	struct Modal {
		std::string title{};
		std::string text{};
	};

	auto make_window(std::string_view build_version) -> kvf::UniqueWindow;

	template <std::derived_from<Scene> T>
	void add_factory(klib::CString name);

	void draw_menu();
	void draw_error_modal() const;

	UniqueWindow m_window;
	RenderDevice m_device;
	std::string_view m_assets_dir;
	std::vector<Factory> m_factories{};
	Factory* m_current_factory{};
	Modal m_modal{};

	std::unique_ptr<Scene> m_scene;

	DeviceBlock m_blocker;
};
} // namespace kvf::example
