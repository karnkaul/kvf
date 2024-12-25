#include <imgui.h>
#include <klib/args/parse.hpp>
#include <klib/assert.hpp>
#include <klib/fixed_string.hpp>
#include <klib/log.hpp>
#include <klib/version_str.hpp>
#include <kvf/build_version.hpp>
#include <kvf/device_block.hpp>
#include <kvf/render_device.hpp>
#include <kvf/render_pass.hpp>
#include <kvf/util.hpp>
#include <kvf/window.hpp>
#include <filesystem>
#include <print>

namespace {
namespace fs = std::filesystem;

struct ShaderLoader {
	vk::Device device{};
	fs::path dir{};
	std::vector<std::uint32_t> spir_v{};

	auto load(std::string_view const uri) -> vk::UniqueShaderModule {
		auto const path = dir / uri;
		auto const result = kvf::util::spirv_from_file(spir_v, path.string().c_str());
		if (result != kvf::IoResult::Success) { throw std::runtime_error{std::format("Failed to load shader: {}", path.generic_string())}; }
		auto smci = vk::ShaderModuleCreateInfo{};
		smci.setCode(spir_v);
		return device.createShaderModuleUnique(smci);
	}
};

struct App {
	explicit App(std::string_view const build_version, std::string_view const assets_dir)
		: m_assets_dir(assets_dir), m_window(make_window(build_version)), m_device(m_window.get()), m_color_pass(&m_device, vk::SampleCountFlagBits::e2) {}

	void run() {
		m_device_blocker = m_device.get_device();

		m_color_pass.set_color_target();
		m_color_pass.set_depth_target();
		m_color_pass.clear_color = vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 1.0f}};

		create_pipeline();

		while (!m_device.is_window_closing()) {
			auto command_buffer = m_device.next_frame();
			KLIB_ASSERT(command_buffer);

			ImGui::ShowDemoWindow();

			static constexpr auto upscale_v = 2.0f;
			auto const framebuffer_extent = kvf::util::scale_extent(m_device.get_framebuffer_extent(), upscale_v);
			m_color_pass.begin_render(command_buffer, framebuffer_extent);

			m_color_pass.bind_pipeline(*m_pipeline);
			command_buffer.draw(3, 1, 0, 0);

			m_color_pass.end_render();

			m_device.render(m_color_pass.render_target());
		}
	}

  private:
	auto make_window(std::string_view const build_version) -> kvf::UniqueWindow {
		auto const title = klib::FixedString{"kvf example [{}]", build_version};
		auto ret = kvf::create_window(title.c_str(), 1280, 720);
		glfwSetWindowUserPointer(ret.get(), this);
		glfwSetKeyCallback(ret.get(), [](GLFWwindow* w, int const key, int const /*scancode*/, int action, int const mods) {
			auto& self = *static_cast<App*>(glfwGetWindowUserPointer(w));
			if (action == GLFW_RELEASE && mods == 0) {
				if (key == GLFW_KEY_ESCAPE) { self.m_device.set_window_closing(true); }
				if (key == GLFW_KEY_I) { self.m_device.set_render_imgui(!self.m_device.get_render_imgui()); }
				if (key == GLFW_KEY_P) { self.switch_present_mode(); }
			}
		});
		return ret;
	}

	void create_pipeline() {
		auto loader = ShaderLoader{
			.device = m_device.get_device(),
			.dir = m_assets_dir,
		};
		auto const vertex_shader = loader.load("shader.vert");
		auto const fragment_shader = loader.load("shader.frag");

		m_pipeline_layout = m_device.get_device().createPipelineLayoutUnique({});
		auto const pipeline_state = kvf::PipelineState{
			.vertex_attributes = {},
			.vertex_bindings = {},
			.vertex_shader = *vertex_shader,
			.fragment_shader = *fragment_shader,
		};
		m_pipeline = m_color_pass.create_pipeline(*m_pipeline_layout, pipeline_state);
		if (!m_pipeline) { throw std::runtime_error{"Failed to create Vulkan Pipeline"}; }
	}

	void switch_present_mode() {
		auto const modes = m_device.get_supported_present_modes();
		auto const current = m_device.get_present_mode();
		auto it = std::ranges::find(modes, current);
		KLIB_ASSERT(it != modes.end());
		std::advance(it, 1);
		if (it == modes.end()) { it = modes.begin(); }
		m_device.set_present_mode(*it);
	}

	std::string_view m_assets_dir{};

	kvf::UniqueWindow m_window{};
	kvf::RenderDevice m_device;

	kvf::RenderPass m_color_pass;

	vk::UniquePipelineLayout m_pipeline_layout{};
	vk::UniquePipeline m_pipeline{};

	kvf::DeviceBlock m_device_blocker{};
};
} // namespace

auto main(int argc, char** argv) -> int {
	auto const log_file = klib::log::File{"kvf-example.log"};
	try {
		auto assets_dir = std::string_view{"."};
		auto const args = std::array{
			klib::args::option(assets_dir, "a,assets", "example assets directory"),
		};
		auto const build_version = klib::to_string(kvf::build_version_v);
		auto const parse_info = klib::args::ParseInfo{.version = build_version};
		auto const parse_result = klib::args::parse(parse_info, args, argc, argv);
		if (parse_result.early_return()) { return parse_result.get_return_code(); }
		klib::log::info("kvf", "Using assets directory: {}", assets_dir);
		App{build_version, assets_dir}.run();
	} catch (std::exception const& e) {
		klib::log::error("PANIC", "{}", e.what());
		return EXIT_FAILURE;
	} catch (...) {
		klib::log::error("PANIC", "Unknown");
		return EXIT_FAILURE;
	}
}
