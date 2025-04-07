#pragma once
#include <kvf/render_api.hpp>
#include <vulkan/vulkan.hpp>
#include <chrono>
#include <gsl/pointers>

using namespace std::chrono_literals;

namespace kvf {
class CommandBuffer {
  public:
	static constexpr auto timeout_v{5s};

	explicit CommandBuffer(gsl::not_null<IRenderApi const*> api);

	[[nodiscard]] auto get() const -> vk::CommandBuffer { return m_cmd; }

	auto submit_and_wait(std::chrono::seconds timeout = timeout_v) -> bool;

	operator vk::CommandBuffer() const { return get(); }

  private:
	IRenderApi const* m_api;

	vk::UniqueCommandPool m_pool{};
	vk::CommandBuffer m_cmd{};
};
} // namespace kvf
