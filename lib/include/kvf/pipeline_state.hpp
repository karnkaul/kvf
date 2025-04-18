#pragma once
#include <klib/enum_ops.hpp>
#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <span>

namespace kvf {
enum class PipelineFlag : std::int8_t;
}

namespace klib {
template <>
inline constexpr auto enable_enum_ops_v<kvf::PipelineFlag> = true;
}

namespace kvf {
enum class PipelineFlag : std::int8_t {
	None = 0,
	AlphaBlend = 1 << 0,
	DepthTest = 1 << 1,
};

struct PipelineState {
	using Flag = PipelineFlag;

	[[nodiscard]] static constexpr auto default_flags() -> Flag { return Flag::AlphaBlend | Flag::DepthTest; }

	std::span<vk::VertexInputBindingDescription const> vertex_bindings;
	std::span<vk::VertexInputAttributeDescription const> vertex_attributes;
	vk::ShaderModule vertex_shader;
	vk::ShaderModule fragment_shader;

	vk::PrimitiveTopology topology{vk::PrimitiveTopology::eTriangleList};
	vk::PolygonMode polygon_mode{vk::PolygonMode::eFill};
	vk::CullModeFlags cull_mode{vk::CullModeFlagBits::eNone};
	vk::CompareOp depth_compare{vk::CompareOp::eLess};
	Flag flags{default_flags()};
};

struct PipelineFormat {
	vk::SampleCountFlagBits samples{};
	vk::Format color{};
	vk::Format depth{};
};
} // namespace kvf
