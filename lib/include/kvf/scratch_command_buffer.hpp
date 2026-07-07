#pragma once
#include "kvf/kvf_fwd.hpp"
#include <vulkan/vulkan.hpp>
#include <chrono>
#include <gsl/pointers>

using namespace std::chrono_literals;

namespace kvf {
class ScratchCommandBuffer {
  public:
	static constexpr auto timeout_v{5s};

	explicit ScratchCommandBuffer(gsl::not_null<IRenderDevice*> render_device);

	[[nodiscard]] auto get_render_device() const -> IRenderDevice& { return *m_render_device; }

	[[nodiscard]] auto get() const -> vk::CommandBuffer { return m_cmd; }

	auto submit_and_wait(std::chrono::seconds timeout = timeout_v) -> bool;

	operator vk::CommandBuffer() const { return get(); }

  private:
	gsl::not_null<IRenderDevice*> m_render_device;

	vk::UniqueCommandPool m_pool{};
	vk::CommandBuffer m_cmd{};
};
} // namespace kvf
