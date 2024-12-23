#include <imgui.h>
#include <klib/args/parse.hpp>
#include <klib/fixed_string.hpp>
#include <klib/log.hpp>
#include <klib/version_str.hpp>
#include <kvf/build_version.hpp>
#include <kvf/render_device.hpp>
#include <kvf/render_pass.hpp>
#include <kvf/util.hpp>
#include <kvf/window.hpp>
#include <filesystem>
#include <fstream>
#include <print>

namespace {
namespace fs = std::filesystem;

struct ShaderLoader {
	vk::Device device{};
	fs::path dir{};
	std::vector<std::uint32_t> spir_v{};

	auto load(std::string_view const uri) -> vk::UniqueShaderModule {
		auto const path = dir / uri;
		if (!fs::is_regular_file(path)) { throw std::runtime_error{std::format("Invalid shader path: {}", path.generic_string())}; }
		auto file = std::ifstream{path, std::ios::binary | std::ios::ate};
		if (!file.is_open()) { throw std::runtime_error{std::format("Failed to open SPIR-V file: {}", uri)}; }
		auto const size = file.tellg();
		if (std::size_t(size) % sizeof(std::uint32_t) != 0) { throw std::runtime_error{std::format("Invalid SPIR-V: {}, size: {}B", uri, std::size_t(size))}; }
		file.seekg(0, std::ios::beg);
		spir_v.resize(std::size_t(size) / sizeof(std::uint32_t));
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
		file.read(reinterpret_cast<char*>(spir_v.data()), size);
		return kvf::util::create_shader_module(device, spir_v);
	}
};

struct App {
	explicit App(std::string_view const build_version, std::string_view const shader_dir)
		: m_shader_dir(shader_dir), m_window(make_window(build_version)), m_device(m_window.get()), m_color_pass(&m_device, vk::SampleCountFlagBits::e4) {}

	void run() {
		m_color_pass.set_color_target();
		m_color_pass.set_depth_target();
		m_color_pass.clear_color = vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 1.0f}};

		create_pipeline();

		while (glfwWindowShouldClose(m_window.get()) != GLFW_TRUE) {
			auto command_buffer = m_device.next_frame();

			ImGui::ShowDemoWindow();

			auto const framebuffer_extent = m_device.get_framebuffer_extent();
			m_color_pass.begin_render(command_buffer, framebuffer_extent);

			command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_pipeline);
			auto const viewport = vk::Viewport{0.0f, float(framebuffer_extent.height), float(framebuffer_extent.width), -float(framebuffer_extent.height)};
			auto const scissor = vk::Rect2D{{}, framebuffer_extent};
			command_buffer.setViewport(0, viewport);
			command_buffer.setScissor(0, scissor);
			command_buffer.draw(3, 1, 0, 0);
			m_color_pass.end_render();

			m_device.render(m_color_pass.render_target());
		}

		m_device.get_device().waitIdle();
	}

  private:
	auto make_window(std::string_view const build_version) -> kvf::UniqueWindow {
		auto const title = klib::FixedString{"kvf example [{}]", build_version};
		auto ret = kvf::create_window(title.c_str(), 1280, 720);
		glfwSetWindowUserPointer(ret.get(), this);
		glfwSetKeyCallback(ret.get(), [](GLFWwindow* w, int const key, int const /*scancode*/, int action, int const mods) {
			if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE && mods == 0) { glfwSetWindowShouldClose(w, GLFW_TRUE); }
			if (key == GLFW_KEY_I && action == GLFW_RELEASE && mods == 0) {
				auto& self = *static_cast<App*>(glfwGetWindowUserPointer(w));
				self.m_device.set_render_imgui(!self.m_device.get_render_imgui());
			}
		});
		return ret;
	}

	void create_pipeline() {
		auto loader = ShaderLoader{
			.device = m_device.get_device(),
			.dir = m_shader_dir,
		};
		auto const vertex_shader = loader.load("shader.vert");
		auto const fragment_shader = loader.load("shader.frag");

		m_pipeline_layout = m_device.get_device().createPipelineLayoutUnique({});
		auto const pipeline_state = kvf::PipelineState{
			.vertex_attributes = {},
			.vertex_bindings = {},
			.vertex_shader = *vertex_shader,
			.fragment_shader = *fragment_shader,
			.color_format = m_color_pass.get_color_format(),
			.depth_format = m_color_pass.get_depth_format(),
			.samples = m_color_pass.get_samples(),
		};
		m_pipeline = kvf::util::create_pipeline(m_device.get_device(), *m_pipeline_layout, pipeline_state);
		if (!m_pipeline) { throw std::runtime_error{"Failed to create Vulkan Pipeline"}; }
	}

	std::string_view m_shader_dir{};

	kvf::UniqueWindow m_window{};
	kvf::RenderDevice m_device;

	kvf::RenderPass m_color_pass;

	vk::UniquePipelineLayout m_pipeline_layout{};
	vk::UniquePipeline m_pipeline{};
};
} // namespace

auto main(int argc, char** argv) -> int {
	auto const log_file = klib::log::File{"kvf-example.log"};
	try {
		auto shader_dir = std::string_view{"."};
		auto const args = std::array{
			klib::args::positional(shader_dir, klib::args::ArgType::Optional, "SHADER_DIR"),
		};
		auto const build_version = klib::to_string(kvf::build_version_v);
		auto const parse_info = klib::args::ParseInfo{.version = build_version};
		auto const parse_result = klib::args::parse(parse_info, args, argc, argv);
		if (parse_result.early_return()) { return parse_result.get_return_code(); }
		auto app = App{build_version, shader_dir};
		app.run();
	} catch (std::exception const& e) {
		klib::log::error("PANIC", "{}", e.what());
		return EXIT_FAILURE;
	} catch (...) {
		klib::log::error("PANIC", "Unknown");
		return EXIT_FAILURE;
	}
}
