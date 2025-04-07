#include <kvf/error.hpp>
#include <kvf/util.hpp>
#include <shader_loader.hpp>
#include <filesystem>

namespace kvf::example {
namespace fs = std::filesystem;

auto ShaderLoader::load(std::string_view const uri) -> vk::UniqueShaderModule {
	auto const path = fs::path{m_dir} / uri;
	if (!kvf::util::spirv_from_file(m_spir_v, path.string().c_str())) { throw Error{std::format("Failed to load shader: {}", path.generic_string())}; }
	auto smci = vk::ShaderModuleCreateInfo{};
	smci.setCode(m_spir_v);
	return m_device.createShaderModuleUnique(smci);
}
} // namespace kvf::example
