#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <kvf/error.hpp>
#include <kvf/util.hpp>
#include <log.hpp>
#include <scenes/sprite.hpp>
#include <shader_loader.hpp>
#include <ranges>

namespace kvf::example {
namespace {
struct Vertex {
	glm::vec2 position{};
	glm::vec2 uv{};
};

struct Quad {
	static constexpr std::uint32_t vertex_count_v{4};
	static constexpr std::uint32_t index_count_v{6};

	std::array<Vertex, vertex_count_v> vertices{
		Vertex{.uv = {0.0f, 1.0f}},
		Vertex{.uv = {1.0f, 1.0f}},
		Vertex{.uv = {1.0f, 0.0f}},
		Vertex{.uv = {0.0f, 0.0f}},
	};
	std::array<std::uint32_t, index_count_v> indices{0, 1, 2, 2, 3, 0};

	void resize(glm::vec2 const size) {
		auto const half_size = 0.5f * size;
		vertices[0].position = {-half_size.x, -half_size.y};
		vertices[1].position = {half_size.x, -half_size.y};
		vertices[2].position = {half_size.x, half_size.y};
		vertices[3].position = {-half_size.x, half_size.y};
	}
};

[[nodiscard]] auto vbo_info() {
	return vma::BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer,
		.type = vma::BufferType::Device,
	};
}

[[nodiscard]] auto ubo_info() {
	return vma::BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eUniformBuffer,
		.type = vma::BufferType::Host,
	};
}

[[nodiscard]] auto ssbo_info() {
	return vma::BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eStorageBuffer,
		.type = vma::BufferType::Host,
	};
}
} // namespace

Sprite::Sprite(gsl::not_null<RenderDevice*> device, std::string_view assets_dir)
	: Scene(device, assets_dir), m_color_pass(device, vk::SampleCountFlagBits::e2), m_vbo(device, vbo_info(), sizeof(Quad)),
	  m_ubo(device, ubo_info(), sizeof(glm::mat4)), m_ssbo(device, ssbo_info()) {
	m_color_pass.set_color_target().set_depth_target();
	m_color_pass.clear_color = vk::ClearColorValue{std::array{0.05f, 0.05f, 0.05f, 1.0f}};

	create_set_layouts();
	create_pipeline_layout();
	create_pipeline();

	create_descriptor_pools();

	write_vbo();
}

void Sprite::update(vk::CommandBuffer const command_buffer) {
	auto const frame_index = std::size_t(get_device().get_frame_index());
	auto const descriptor_pool = *m_descriptor_pools.at(frame_index);
	get_device().get_device().resetDescriptorPool(descriptor_pool);
	auto descriptor_sets = std::array<vk::DescriptorSet, 2>{};
	auto dsai = vk::DescriptorSetAllocateInfo{};
	dsai.setDescriptorPool(descriptor_pool).setSetLayouts(m_set_layouts).setDescriptorSetCount(2);
	auto const result = get_device().get_device().allocateDescriptorSets(&dsai, descriptor_sets.data());
	if (result != vk::Result::eSuccess) {
		log::warn("Failed to allocate Descriptor Sets");
		return;
	}

	auto const extent = get_device().get_framebuffer_extent();

	m_color_pass.begin_render(command_buffer, extent);

	m_color_pass.bind_pipeline(*m_pipeline);

	write_descriptor_sets(descriptor_sets, util::to_glm_vec(extent));
	command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_pipeline_layout, 0, descriptor_sets, {});

	command_buffer.bindVertexBuffers(0, m_vbo.get_buffer(), vk::DeviceSize{0});
	command_buffer.bindIndexBuffer(m_vbo.get_buffer(), m_index_offset, vk::IndexType::eUint32);
	command_buffer.drawIndexed(Quad::index_count_v, 1, 0, 0, 0);

	m_color_pass.end_render();
}

auto Sprite::get_render_target() const -> RenderTarget { return m_color_pass.render_target(); }

void Sprite::create_set_layouts() {
	auto set_0_bindings = std::array<vk::DescriptorSetLayoutBinding, 1>{};
	set_0_bindings[0]
		.setBinding(0)
		.setDescriptorCount(1)
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setStageFlags(vk::ShaderStageFlagBits::eAllGraphics);
	auto dslci = vk::DescriptorSetLayoutCreateInfo{};
	dslci.setBindings(set_0_bindings);
	m_set_layout_storage[0] = get_device().get_device().createDescriptorSetLayoutUnique(dslci);

	auto set_1_bindings = std::array<vk::DescriptorSetLayoutBinding, 2>{};
	set_1_bindings[0]
		.setBinding(0)
		.setDescriptorCount(1)
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setStageFlags(vk::ShaderStageFlagBits::eAllGraphics);
	set_1_bindings[1]
		.setBinding(1)
		.setDescriptorCount(1)
		.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
		.setStageFlags(set_1_bindings[0].stageFlags);
	dslci.setBindings(set_1_bindings);
	m_set_layout_storage[1] = get_device().get_device().createDescriptorSetLayoutUnique(dslci);

	for (auto [storage, set_layout] : std::ranges::zip_view(m_set_layout_storage, m_set_layouts)) { set_layout = *storage; }
}

void Sprite::create_pipeline_layout() {
	auto const set_layouts = std::array{*m_set_layout_storage[0], *m_set_layout_storage[1]};
	auto plci = vk::PipelineLayoutCreateInfo{};
	plci.setSetLayouts(set_layouts);
	m_pipeline_layout = get_device().get_device().createPipelineLayoutUnique(plci);
}

void Sprite::create_pipeline() {
	auto loader = ShaderLoader{get_device().get_device(), get_assets_dir()};
	auto const vertex_shader = loader.load("sprite.vert");
	auto const fragment_shader = loader.load("sprite.frag");

	auto bindings = std::array<vk::VertexInputBindingDescription, 1>{};
	bindings[0].setBinding(0).setInputRate(vk::VertexInputRate::eVertex).setStride(sizeof(Vertex));

	auto attributes = std::array<vk::VertexInputAttributeDescription, 2>{};
	attributes[0].setBinding(0).setLocation(0).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, position));
	attributes[1].setBinding(0).setLocation(1).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, uv));

	auto const pipeline_state = kvf::PipelineState{
		.vertex_bindings = bindings,
		.vertex_attributes = attributes,
		.vertex_shader = *vertex_shader,
		.fragment_shader = *fragment_shader,
	};
	m_pipeline = m_color_pass.create_pipeline(*m_pipeline_layout, pipeline_state);
	if (!m_pipeline) { throw Error{"Failed to create Vulkan Pipeline"}; }
}

void Sprite::create_descriptor_pools() {
	auto const pool_sizes = std::array{
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 2},
		vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1},
	};
	auto dpci = vk::DescriptorPoolCreateInfo{};
	dpci.setPoolSizes(pool_sizes).setMaxSets(buffering_v * pool_sizes.size());
	for (auto& descriptor_pool : m_descriptor_pools) { descriptor_pool = get_device().get_device().createDescriptorPoolUnique(dpci); }
}

void Sprite::write_vbo() {
	auto quad = Quad{};
	quad.resize(glm::vec2{200.0f});

	auto const vertices = std::span{quad.vertices};
	auto buffer_write = BufferWrite{.ptr = vertices.data(), .size = vertices.size_bytes()};
	if (!util::overwrite(m_vbo, buffer_write, 0)) { throw Error{"Failed to write vertices to Buffer"}; }

	m_index_offset = buffer_write.size;
	auto const indices = std::span{quad.indices};
	buffer_write = BufferWrite{.ptr = indices.data(), .size = indices.size_bytes()};
	if (!util::overwrite(m_vbo, buffer_write, m_index_offset)) { throw Error{"Failed to write indices to Buffer"}; }
}

void Sprite::write_descriptor_sets(std::span<vk::DescriptorSet const, 2> sets, glm::vec2 const extent) {
	auto const half_extent = 0.5f * extent;
	auto const projection = glm::ortho(-half_extent.x, half_extent.x, -half_extent.y, half_extent.y);
	if (!util::write_to(m_ubo, {&projection, sizeof(projection)})) { throw Error{"Failed to write to Uniform Buffer"}; }

	auto wds = std::array<vk::WriteDescriptorSet, 1>{};
	auto dbi = vk::DescriptorBufferInfo{};
	dbi.setBuffer(m_ubo.get_buffer()).setRange(m_ubo.get_size());
	wds[0].setDescriptorCount(1).setDescriptorType(vk::DescriptorType::eUniformBuffer).setBufferInfo(dbi).setDstSet(sets[0]).setDstBinding(0);

	get_device().get_device().updateDescriptorSets(wds, {});
}
} // namespace kvf::example
