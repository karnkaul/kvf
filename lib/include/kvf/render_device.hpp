#pragma once
#include <klib/constants.hpp>
#include <klib/version.hpp>
#include <kvf/buffering.hpp>
#include <kvf/wsi.hpp>
#include <vulkan/vulkan.hpp>
#include <gsl/pointers>
#include <memory>
#include <optional>

namespace kvf {
enum RenderDeviceFlag : int {
	None = 0,
	ValidationLayers = 1 << 0,
};
using RenderDeviceFlags = std::underlying_type_t<RenderDeviceFlag>;

struct RenderDeviceCreateInfo {
	RenderDeviceFlags flags{};
	std::optional<vk::PresentModeKHR> present_mode{};
};

class RenderDevice {
  public:
	static constexpr auto vk_api_version_v = klib::Version{.major = 1, .minor = 3};

	using Flags = RenderDeviceFlags;

	static constexpr auto default_flags() -> Flags {
		auto ret = Flags{};
		if constexpr (klib::debug_v) { ret |= RenderDeviceFlag::ValidationLayers; }
		return ret;
	}

	explicit RenderDevice(gsl::not_null<IWsi const*> wsi, Flags flags = default_flags());

	[[nodiscard]] auto get_loader_api_version() const -> klib::Version;
	[[nodiscard]] auto get_instance() const -> vk::Instance;
	[[nodiscard]] auto get_surface() const -> vk::SurfaceKHR;
	[[nodiscard]] auto get_gpu() const -> Gpu const&;
	[[nodiscard]] auto get_device() const -> vk::Device;
	[[nodiscard]] auto get_queue_family() const -> std::uint32_t;

	[[nodiscard]] auto get_present_mode() const -> vk::PresentModeKHR;
	[[nodiscard]] auto get_supported_present_modes() const -> std::span<vk::PresentModeKHR const>;
	auto request_present_mode(vk::PresentModeKHR desired) -> bool;

	auto acquire_next_image() -> bool;
	void present_acquired_image();
	void temp_render();

  private:
	struct Impl;
	struct Deleter {
		void operator()(Impl* ptr) const noexcept;
	};

	std::unique_ptr<Impl, Deleter> m_impl{};
};
} // namespace kvf
