#pragma once
#include <klib/enum_ops.hpp>
#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <span>

namespace kvf {
enum class PipelineFlag : std::uint8_t;
} // namespace kvf

template <>
inline constexpr auto klib::enable_enum_ops_v<kvf::PipelineFlag> = true;

namespace kvf {
enum class PipelineFlag : std::uint8_t {
	None = 0,
	DepthTest = 1 << 0,
	DepthWrite = 1 << 1,
};

struct PipelineState {
	using Flag = PipelineFlag;

	[[nodiscard]] static constexpr auto default_flags() -> Flag { return Flag::DepthTest | Flag::DepthWrite; }

	[[nodiscard]] static constexpr auto default_blend_state() -> vk::PipelineColorBlendAttachmentState {
		auto ret = vk::PipelineColorBlendAttachmentState{};
		using CCF = vk::ColorComponentFlagBits;
		ret.setColorWriteMask(CCF::eR | CCF::eG | CCF::eB | CCF::eA)
			.setBlendEnable(vk::True)
			.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
			.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
			.setColorBlendOp(vk::BlendOp::eAdd)
			.setSrcAlphaBlendFactor(vk::BlendFactor::eZero)
			.setDstAlphaBlendFactor(vk::BlendFactor::eOne)
			.setAlphaBlendOp(vk::BlendOp::eAdd);
		return ret;
	}

	std::span<vk::VertexInputBindingDescription const> vertex_bindings;
	std::span<vk::VertexInputAttributeDescription const> vertex_attributes;
	vk::ShaderModule vertex_shader;
	vk::ShaderModule fragment_shader;

	vk::PrimitiveTopology topology{vk::PrimitiveTopology::eTriangleList};
	vk::PolygonMode polygon_mode{vk::PolygonMode::eFill};
	vk::CullModeFlags cull_mode{vk::CullModeFlagBits::eNone};
	vk::PipelineColorBlendAttachmentState blend_state{default_blend_state()};
	vk::CompareOp depth_compare{vk::CompareOp::eLess};
	Flag flags{default_flags()};
};

struct PipelineFormat {
	vk::SampleCountFlagBits samples{};
	vk::Format color{};
	vk::Format depth{};
};
} // namespace kvf
