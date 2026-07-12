#include "shader_loader.hpp"
#include "klib/file_io.hpp"
#include "kvf/panic.hpp"
#include <filesystem>

namespace kvf::example {
namespace fs = std::filesystem;

auto ShaderLoader::load_module(std::string_view const uri) const -> vk::UniqueShaderModule {
	auto const spir_v = load_spir_v(uri);
	auto smci = vk::ShaderModuleCreateInfo{};
	smci.setCode(spir_v);
	return m_device.createShaderModuleUnique(smci);
}

auto ShaderLoader::load_spir_v(std::string_view uri) const -> std::vector<std::uint32_t> {
	auto const path = fs::path{m_dir} / uri;
	auto ret = std::vector<std::uint32_t>{};
	if (!klib::copy_file_bytes_to(ret, path.string().c_str())) { throw Panic{std::format("Failed to load shader: {}", path.generic_string())}; }
	return ret;
}
} // namespace kvf::example
