#pragma once
#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <vector>

namespace kvf::example {
class ShaderLoader {
  public:
	explicit ShaderLoader(vk::Device device, std::string_view dir) : m_device(device), m_dir(dir) {}

	[[nodiscard]] auto load_module(std::string_view uri) const -> vk::UniqueShaderModule;
	[[nodiscard]] auto load_spir_v(std::string_view uri) const -> std::vector<std::uint32_t>;

  private:
	vk::Device m_device{};
	std::string_view m_dir{};
};
} // namespace kvf::example
