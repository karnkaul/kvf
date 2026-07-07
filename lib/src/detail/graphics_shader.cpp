#include "kvf/graphics_shader.hpp"
#include "kvf/render_device.hpp"
#include "log.hpp"

namespace kvf {
namespace detail {
namespace {
class GraphicsShader : public IGraphicsShader {
  public:
	explicit GraphicsShader(GraphicsShaderInput const& input, vk::UniqueShaderEXT vertex, vk::UniqueShaderEXT fragment)
		: m_input(input), m_vertex(std::move(vertex)), m_fragment(std::move(fragment)) {}

  private:
	[[nodiscard]] auto get_input() const -> GraphicsShaderInput const& final { return m_input; }
	[[nodiscard]] auto get_vertex() const -> vk::ShaderEXT final { return *m_vertex; }
	[[nodiscard]] auto get_fragment() const -> vk::ShaderEXT final { return *m_fragment; }

	GraphicsShaderInput m_input{};
	vk::UniqueShaderEXT m_vertex{};
	vk::UniqueShaderEXT m_fragment{};
};
} // namespace
} // namespace detail

auto IGraphicsShader::create(gsl::not_null<kvf::IRenderDevice const*> render_device, CreateInfo const& create_info) -> std::unique_ptr<IGraphicsShader> {
	if ((render_device->get_flags() & RenderDeviceFlag::ShaderObjectFeature) != RenderDeviceFlag::ShaderObjectFeature) {
		log.warn("Attempt to create ShaderEXT objects without ShaderObjectFeature");
		return {};
	}

	auto const create_shader_ci = [&create_info](std::span<std::uint32_t const> spirv) {
		auto ret = vk::ShaderCreateInfoEXT{};
		ret.setCodeSize(spirv.size_bytes())
			.setPCode(spirv.data())
			.setSetLayouts(create_info.set_layouts)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setPName("main");
		ret.flags |= vk::ShaderCreateFlagBitsEXT::eLinkStage;
		return ret;
	};

	auto shader_cis = std::array{
		create_shader_ci(create_info.code.vertex),
		create_shader_ci(create_info.code.fragment),
	};
	shader_cis[0].setStage(vk::ShaderStageFlagBits::eVertex).setNextStage(vk::ShaderStageFlagBits::eFragment);
	shader_cis[1].setStage(vk::ShaderStageFlagBits::eFragment);

	auto const device = render_device->get_device();
	auto shaders = std::array<vk::ShaderEXT, 2>{};
	auto result = device.createShadersEXT(std::uint32_t(shader_cis.size()), shader_cis.data(), nullptr, shaders.data());
	if (result != vk::Result::eSuccess) {
		log.error("Failed to create Vulkan ShaderEXT objects");
		return {};
	}

	return std::make_unique<detail::GraphicsShader>(create_info.input, vk::UniqueShaderEXT{shaders[0], device}, vk::UniqueShaderEXT{shaders[1], device});
}
} // namespace kvf
