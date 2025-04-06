#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <klib/random.hpp>
#include <kvf/error.hpp>
#include <kvf/image_bitmap.hpp>
#include <kvf/util.hpp>
#include <log.hpp>
#include <scenes/sprite.hpp>
#include <shader_loader.hpp>
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

[[nodiscard]] constexpr auto buffer_info(vk::BufferUsageFlags const usage) {
	return vma::BufferCreateInfo{
		.usage = usage,
		.type = vma::BufferType::Device,
	};
}

[[nodiscard]] constexpr auto vbo_info() { return buffer_info(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer); }
[[nodiscard]] constexpr auto ubo_info() { return buffer_info(vk::BufferUsageFlagBits::eUniformBuffer); }
[[nodiscard]] constexpr auto ssbo_info() { return buffer_info(vk::BufferUsageFlagBits::eStorageBuffer); }
[[nodiscard]] constexpr auto texture_info() { return vma::ImageCreateInfo{.format = vk::Format::eR8G8B8A8Srgb, .flags = vma::ImageFlag::MipMapped}; }
} // namespace

Sprite::Sprite(gsl::not_null<RenderDevice*> device, std::string_view assets_dir)
	: Scene(device, assets_dir), m_color_pass(device, vk::SampleCountFlagBits::e2), m_vbo(device, vbo_info(), sizeof(Quad)), m_texture(device, texture_info()) {
	for (auto& ubo : m_ubos) { ubo = vma::Buffer{device, ubo_info(), sizeof(glm::mat4)}; }
	for (auto& ssbo : m_ssbos) { ssbo = vma::Buffer{device, ssbo_info()}; }

	m_color_pass.set_color_target().set_depth_target();
	m_color_pass.clear_color = Color{glm::vec4{0.1f, 0.1f, 0.1f, 1.0f}}.to_linear();

	create_set_layouts();
	create_pipeline_layout();
	create_pipeline();
	create_texture();

	write_vbo();

	create_instances();
}

void Sprite::update(vk::CommandBuffer const command_buffer) {
	for (auto& instance : m_instances) { instance.rotation += instance.degrees_per_sec * get_dt().count(); }

	auto const extent = get_render_device().get_framebuffer_extent();

	m_color_pass.begin_render(command_buffer, extent);

	m_color_pass.bind_pipeline(*m_pipeline);

	auto descriptor_sets = std::array<vk::DescriptorSet, 2>{};
	if (get_render_device().allocate_sets(descriptor_sets, m_set_layouts)) {
		write_descriptor_sets(descriptor_sets, util::to_glm_vec(extent));
		command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_pipeline_layout, 0, descriptor_sets, {});

		command_buffer.bindVertexBuffers(0, m_vbo.get_buffer(), vk::DeviceSize{0});
		command_buffer.bindIndexBuffer(m_vbo.get_buffer(), m_index_offset, vk::IndexType::eUint32);
		command_buffer.drawIndexed(Quad::index_count_v, std::uint32_t(m_instances.size()), 0, 0, 0);
	}

	m_color_pass.end_render();
}

auto Sprite::get_render_target() const -> RenderTarget { return m_color_pass.render_target(); }

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

void Sprite::create_pipeline() {
	auto loader = ShaderLoader{get_render_device().get_device(), get_assets_dir()};
	auto const vertex_shader = loader.load("sprite.vert");
	auto const fragment_shader = loader.load("sprite.frag");

	auto bindings = std::array<vk::VertexInputBindingDescription, 1>{};
	bindings[0].setBinding(0).setInputRate(vk::VertexInputRate::eVertex).setStride(sizeof(Vertex));

	auto attributes = std::array<vk::VertexInputAttributeDescription, 2>{};
	attributes[0].setBinding(0).setLocation(0).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, position));
	attributes[1].setBinding(0).setLocation(1).setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, uv));

	auto const pipeline_state = PipelineState{
		.vertex_bindings = bindings,
		.vertex_attributes = attributes,
		.vertex_shader = *vertex_shader,
		.fragment_shader = *fragment_shader,
		.flags = PipelineFlag::AlphaBlend,
	};
	m_pipeline = m_color_pass.create_pipeline(*m_pipeline_layout, pipeline_state);
	if (!m_pipeline) { throw Error{"Failed to create Vulkan Pipeline"}; }
}

void Sprite::create_texture() {
	auto bytes = std::vector<std::byte>{};
	auto const path = (fs::path{get_assets_dir()} / "awesomeface.png").generic_string();
	if (!util::bytes_from_file(bytes, path.c_str())) { throw Error{std::format("Failed to load image: {}", path)}; }
	auto const image = ImageBitmap{bytes};
	if (!image.is_loaded()) { throw Error{"Failed to load image: awesomeface.png"}; }
	if (!m_texture.resize_and_overwrite(image.bitmap())) { throw Error{"Failed to write to Vulkan Image"}; }

	auto const sci = get_render_device().sampler_info(vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear);
	m_sampler = get_render_device().get_device().createSamplerUnique(sci);
}

void Sprite::write_vbo() {
	auto quad = Quad{};
	quad.resize(glm::vec2{100.0f});

	auto const vertices = std::span{quad.vertices};
	if (!m_vbo.write_in_place(vertices)) { throw Error{"Failed to write vertices to Buffer"}; }

	m_index_offset = vertices.size_bytes();
	auto const indices = std::span{quad.indices};
	if (!m_vbo.write_in_place(indices, m_index_offset)) { throw Error{"Failed to write indices to Buffer"}; }
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
	auto const half_extent = 0.5f * extent;
	auto const projection = glm::ortho(-half_extent.x, half_extent.x, -half_extent.y, half_extent.y);
	auto& ubo = m_ubos.at(std::size_t(get_render_device().get_frame_index()));
	if (!ubo.resize_and_overwrite(projection)) { throw Error{"Failed to write to Uniform Buffer"}; }

	auto wds = std::array<vk::WriteDescriptorSet, 3>{};
	auto view_dbi = vk::DescriptorBufferInfo{};
	view_dbi.setBuffer(ubo.get_buffer()).setRange(ubo.get_size());
	wds[0].setDescriptorCount(1).setDescriptorType(vk::DescriptorType::eUniformBuffer).setBufferInfo(view_dbi).setDstSet(sets[0]).setDstBinding(0);

	m_instance_buffer.clear();
	m_instance_buffer.reserve(m_instances.size());
	for (auto const& instance : m_instances) {
		auto const t = glm::translate(glm::mat4{1.0f}, glm::vec3{instance.position, 0.0f});
		auto const r = glm::rotate(glm::mat4{1.0f}, glm::radians(instance.rotation), glm::vec3{0.0f, 0.0f, 1.0f});
		m_instance_buffer.push_back(Std430Instance{.mat_world = t * r, .tint = instance.tint.to_vec4()});
	}
	auto& ssbo = m_ssbos.at(std::size_t(get_render_device().get_frame_index()));
	if (!ssbo.resize_and_overwrite(std::span{m_instance_buffer})) { throw Error{"Failed to write to Storage Buffer"}; }
	auto instances_dbi = vk::DescriptorBufferInfo{};
	instances_dbi.setBuffer(ssbo.get_buffer()).setRange(ssbo.get_size());
	wds[1].setDescriptorCount(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(instances_dbi).setDstSet(sets[1]).setDstBinding(0);

	auto texture_dii = vk::DescriptorImageInfo{};
	texture_dii.setImageView(m_texture.get_view()).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal).setSampler(*m_sampler);
	wds[2].setDescriptorCount(1).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setImageInfo(texture_dii).setDstSet(sets[1]).setDstBinding(1);

	get_render_device().get_device().updateDescriptorSets(wds, {});
}
} // namespace kvf::example
