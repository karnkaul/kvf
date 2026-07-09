#include "scenes/sprite.hpp"
#include "klib/debug/assert.hpp"
#include "klib/random.hpp"
#include "kvf/image_bitmap.hpp"
#include "kvf/panic.hpp"
#include "kvf/util.hpp"
#include "shader_loader.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <imgui.h>
#include <array>
#include <filesystem>
#include <ranges>

namespace kvf::example {
namespace fs = std::filesystem;

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

constexpr auto vbo_ci_v = BufferCreateInfo{
	.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer,
	.type = BufferType::Device,
	.size = sizeof(Quad),
};

constexpr auto buffer_usage_layout_v = std::array<vk::BufferUsageFlags, 2>{
	vk::BufferUsageFlagBits::eUniformBuffer,
	vk::BufferUsageFlagBits::eStorageBuffer,
};
} // namespace

Sprite::Sprite(gsl::not_null<IRenderDevice*> device, std::string_view assets_dir)
	: Scene(device, assets_dir), m_color_pass(IRenderPass::create(device, vk::SampleCountFlagBits::e2)),
	  m_scratch_buffers(IRingBufferAllocator::create(device, buffer_usage_layout_v)), m_vbo(IRenderBuffer::create(device, vbo_ci_v)) {
	m_color_pass->set_color_target();
	m_color_pass->set_depth_target();
	m_color_pass->clear_color = Color{glm::vec4{0.1f, 0.1f, 0.1f, 1.0f}}.to_linear();

	create_set_layouts();
	create_pipeline_layout();
	create_shader();
	create_texture();

	write_vbo();

	create_instances();
}

void Sprite::update(vk::CommandBuffer const command_buffer) {
	for (auto& instance : m_instances) { instance.rotation += instance.degrees_per_sec * get_dt().count(); }

	auto const extent = get_render_device().get_swapchain_image_extent();

	m_color_pass->begin_render(command_buffer, extent);

	auto descriptor_sets = std::array<vk::DescriptorSet, 2>{};
	if (m_color_pass->allocate_sets(descriptor_sets, m_set_layouts)) {
		write_descriptor_sets(descriptor_sets, util::to_glm_vec(extent));
		command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_pipeline_layout, 0, descriptor_sets, {});

		m_color_pass->bind_graphics_shader(*m_shader);

		command_buffer.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);

		command_buffer.bindVertexBuffers(0, m_vbo->get_buffer(), vk::DeviceSize{0});
		command_buffer.bindIndexBuffer(m_vbo->get_buffer(), m_index_offset, vk::IndexType::eUint32);
		command_buffer.drawIndexed(Quad::index_count_v, std::uint32_t(m_instances.size()), 0, 0, 0);
	}

	m_color_pass->end_render();
}

auto Sprite::get_render_target() const -> RenderTarget { return m_color_pass->render_target(); }

void Sprite::create_set_layouts() {
	static constexpr auto stages_v = vk::ShaderStageFlagBits::eAllGraphics;
	auto const set_0 = vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer, 1, stages_v};
	auto dslci = vk::DescriptorSetLayoutCreateInfo{};
	dslci.setBindings(set_0);
	m_set_layout_storage[0] = get_render_device().get_device().createDescriptorSetLayoutUnique(dslci);

	auto const set_1 = std::array{
		vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, stages_v},
		vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eCombinedImageSampler, 1, stages_v},
	};
	dslci.setBindings(set_1);
	m_set_layout_storage[1] = get_render_device().get_device().createDescriptorSetLayoutUnique(dslci);

	for (auto [storage, set_layout] : std::ranges::zip_view(m_set_layout_storage, m_set_layouts)) { set_layout = *storage; }
}

void Sprite::create_pipeline_layout() {
	auto plci = vk::PipelineLayoutCreateInfo{};
	plci.setSetLayouts(m_set_layouts);
	m_pipeline_layout = get_render_device().get_device().createPipelineLayoutUnique(plci);
}

void Sprite::create_shader() {
	auto loader = ShaderLoader{get_render_device().get_device(), get_assets_dir()};

	auto const vert_spir_v = loader.load_spir_v("sprite.vert");
	auto const frag_spir_v = loader.load_spir_v("sprite.frag");
	auto const shader_code = GraphicsShaderCode{
		.vertex = vert_spir_v,
		.fragment = frag_spir_v,
	};
	static constexpr auto input_bindings_v = [] {
		auto ret = std::array<vk::VertexInputBindingDescription2EXT, 1>{};
		ret[0].setBinding(0).setInputRate(vk::VertexInputRate::eVertex).setStride(sizeof(Vertex)).setDivisor(1);
		return ret;
	}();
	static constexpr auto input_attributes_v = [] {
		auto ret = std::array<vk::VertexInputAttributeDescription2EXT, 2>{};
		ret[0].setBinding(0).setLocation(0).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, position));
		ret[1].setBinding(0).setLocation(1).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, uv));
		return ret;
	}();
	static constexpr auto shader_input_v = GraphicsShaderInput{.bindings = input_bindings_v, .attributes = input_attributes_v};

	auto const shader_ci = IGraphicsShader::CreateInfo{
		.code = shader_code,
		.input = shader_input_v,
		.set_layouts = m_set_layouts,
	};
	m_shader = IGraphicsShader::create(&get_render_device(), shader_ci);
}

void Sprite::create_texture() {
	auto bytes = std::vector<std::byte>{};
	auto const path = (fs::path{get_assets_dir()} / "awesomeface.png").generic_string();
	if (!util::bytes_from_file(bytes, path.c_str())) { throw Panic{std::format("Failed to load image: {}", path)}; }
	auto const image = ImageBitmap{bytes};
	if (!image.is_loaded()) { throw Panic{"Failed to load image: awesomeface.png"}; }
	m_texture = IRenderImage::create_texture(&get_render_device(), image.bitmap());

	auto const sci = util::create_sampler_ci(vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear);
	m_sampler = get_render_device().create_sampler(sci);
}

void Sprite::write_vbo() {
	auto quad = Quad{};
	quad.resize(glm::vec2{100.0f});

	auto const vertices = std::span{quad.vertices};
	if (!m_vbo->write_in_place(vertices)) { throw Panic{"Failed to write vertices to Buffer"}; }

	m_index_offset = vertices.size_bytes();
	auto const indices = std::span{quad.indices};
	if (!m_vbo->write_in_place(indices, m_index_offset)) { throw Panic{"Failed to write indices to Buffer"}; }
}

void Sprite::create_instances() {
	static constexpr auto tints_v = std::array{white_v, red_v, green_v};
	auto random_gen = std::mt19937{std::random_device{}()};
	for (int row = -1; row <= 1; ++row) {
		for (int col = -1; col <= 1; ++col) {
			m_instances.push_back(RenderInstance{
				.position = {float(row) * 200.0f, float(col) * 200.0f},
				.degrees_per_sec = klib::random_float(random_gen, -360.0f, 360.0f),
				.tint = tints_v.at(klib::random_index(random_gen, tints_v.size())),
			});
		}
	}
}

void Sprite::write_descriptor_sets(std::span<vk::DescriptorSet const, 2> sets, glm::vec2 const extent) {
	auto const buffers = m_scratch_buffers->allocate_next();
	KLIB_ASSERT(buffers.size() == 2 && buffers[0].get_usage() == vk::BufferUsageFlagBits::eUniformBuffer &&
				buffers[1].get_usage() == vk::BufferUsageFlagBits::eStorageBuffer);

	auto const& view_ubo = buffers[0];
	auto const& instances_ssbo = buffers[1];

	auto wds = std::array<vk::WriteDescriptorSet, 3>{};
	auto const half_extent = 0.5f * extent;
	auto const projection = glm::ortho(-half_extent.x, half_extent.x, -half_extent.y, half_extent.y);
	view_ubo.write(projection);
	auto const view_dbi = view_ubo.descriptor_info();
	wds[0] = util::ubo_write(&view_dbi, sets[0], 0);

	m_instance_buffer.clear();
	m_instance_buffer.reserve(m_instances.size());
	for (auto const& instance : m_instances) {
		auto const t = glm::translate(glm::mat4{1.0f}, glm::vec3{instance.position, 0.0f});
		auto const r = glm::rotate(glm::mat4{1.0f}, glm::radians(instance.rotation), glm::vec3{0.0f, 0.0f, 1.0f});
		m_instance_buffer.push_back(Std430Instance{.mat_world = t * r, .tint = instance.tint.to_vec4()});
	}
	instances_ssbo.write(std::span{m_instance_buffer});
	auto const instances_dbi = instances_ssbo.descriptor_info();
	wds[1] = util::ssbo_write(&instances_dbi, sets[1], 0);

	auto const texture_dii = m_texture->descriptor_info(*m_sampler);
	wds[2] = util::image_write(&texture_dii, sets[1], 1);

	get_render_device().get_device().updateDescriptorSets(wds, {});
}
} // namespace kvf::example
