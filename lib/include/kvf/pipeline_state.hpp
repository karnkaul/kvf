#pragma once
#include <vulkan/vulkan.hpp>
#include <span>

namespace kvf {
struct PipelineState {
	enum : int {
		None = 0,
		AlphaBlend = 1 << 0,
		DepthTest = 1 << 1,
	};
	using Flags = int;

	[[nodiscard]] static constexpr auto default_flags() -> Flags { return AlphaBlend | DepthTest; }

	std::span<vk::VertexInputAttributeDescription const> vertex_attributes;
	std::span<vk::VertexInputBindingDescription const> vertex_bindings;
	vk::ShaderModule vertex_shader;
	vk::ShaderModule fragment_shader;
	vk::Format color_format;
	vk::Format depth_format;
	vk::SampleCountFlagBits samples;

	vk::PrimitiveTopology topology{vk::PrimitiveTopology::eTriangleList};
	vk::PolygonMode polygon_mode{vk::PolygonMode::eFill};
	vk::CullModeFlags cull_mode{vk::CullModeFlagBits::eNone};
	vk::CompareOp depth_compare{vk::CompareOp::eLess};
	Flags flags{default_flags()};
};
} // namespace kvf
