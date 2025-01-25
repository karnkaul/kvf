#pragma once
#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <vector>

namespace kvf::example {
class ShaderLoader {
  public:
	explicit ShaderLoader(vk::Device device, std::string_view dir) : m_device(device), m_dir(dir) {}

	auto load(std::string_view uri) -> vk::UniqueShaderModule;

  private:
	vk::Device m_device{};
	std::string_view m_dir{};
	std::vector<std::uint32_t> m_spir_v{};
};
} // namespace kvf::example
