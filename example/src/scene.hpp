#pragma once
#include <klib/base_types.hpp>
#include <kvf/render_device.hpp>
#include <kvf/time.hpp>

namespace kvf::example {
class Scene : public klib::Polymorphic {
  public:
	explicit Scene(gsl::not_null<RenderDevice*> device, std::string_view assets_dir) : m_device(device), m_assets_dir(assets_dir) {}

  protected:
	struct KeyInput {
		int key{};
		int action{};
		int mods{};
	};

	[[nodiscard]] auto get_render_device() const -> RenderDevice& { return *m_device; }
	[[nodiscard]] auto get_assets_dir() const -> std::string_view { return m_assets_dir; }
	[[nodiscard]] auto get_dt() const -> Seconds { return m_dt; }

	virtual void on_key([[maybe_unused]] KeyInput const& input) {}
	virtual void on_drop([[maybe_unused]] std::span<char const* const> paths) {}
	virtual void update([[maybe_unused]] vk::CommandBuffer command_buffer) {}
	[[nodiscard]] virtual auto get_render_filter() const -> vk::Filter { return vk::Filter::eLinear; }
	[[nodiscard]] virtual auto get_render_target() const -> RenderTarget { return {}; }

	void open_error_modal(std::string message) { m_modal = Modal{.message = std::move(message), .set_open = true}; }

  private:
	struct Modal {
		std::string message{};
		bool set_open{};
	};

	gsl::not_null<RenderDevice*> m_device;
	std::string_view m_assets_dir{};

	Seconds m_dt{};
	Modal m_modal{};

	friend class App;
};
} // namespace kvf::example
