#include "shader_loader.hpp"
#include "kvf/panic.hpp"
#include "kvf/util.hpp"
#include <filesystem>

namespace kvf::example {
namespace fs = std::filesystem;

auto ShaderLoader::load_module(std::string_view const uri) -> vk::UniqueShaderModule {
	auto const bytes = load_bytes(uri);
	auto smci = vk::ShaderModuleCreateInfo{};
	smci.setCode(bytes);
	return m_device.createShaderModuleUnique(smci);
}

auto ShaderLoader::load_bytes(std::string_view uri) -> std::span<std::uint32_t const> {
	auto const path = fs::path{m_dir} / uri;
	if (!kvf::util::spirv_from_file(m_spir_v, path.string().c_str())) { throw Panic{std::format("Failed to load shader: {}", path.generic_string())}; }
	return m_spir_v;
}
} // namespace kvf::example
