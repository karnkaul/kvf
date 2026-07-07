#pragma once
#include "klib/base_types.hpp"
#include "kvf/render_device_fwd.hpp"
#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <gsl/pointers>
#include <memory>
#include <span>

namespace kvf {
struct GraphicsShaderCode {
	std::span<std::uint32_t const> vertex{};
	std::span<std::uint32_t const> fragment{};
};

struct GraphicsShaderInput {
	std::span<vk::VertexInputBindingDescription2EXT const> bindings{};
	std::span<vk::VertexInputAttributeDescription2EXT const> attributes{};
};

struct GraphicsShaderCreateInfo {
	GraphicsShaderCode code{};
	GraphicsShaderInput input{};
	std::span<vk::DescriptorSetLayout const> set_layouts{};
};

class IGraphicsShader : public klib::Polymorphic {
  public:
	using CreateInfo = GraphicsShaderCreateInfo;
	using Input = GraphicsShaderInput;

	[[nodiscard]] static auto create(gsl::not_null<kvf::IRenderDevice const*> render_device, CreateInfo const& create_info) -> std::unique_ptr<IGraphicsShader>;

	[[nodiscard]] virtual auto get_input() const -> GraphicsShaderInput const& = 0;
	[[nodiscard]] virtual auto get_vertex() const -> vk::ShaderEXT = 0;
	[[nodiscard]] virtual auto get_fragment() const -> vk::ShaderEXT = 0;
};
} // namespace kvf
