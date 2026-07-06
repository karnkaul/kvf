#include "app.hpp"
#include "clap/parser.hpp"
#include "klib/log/file.hpp"
#include "kvf/build_version.hpp"
#include "log.hpp"

#include "kvf/util.hpp"
#include <imgui.h>

auto main(int argc, char** argv) -> int {
	auto const log_file = klib::log::File{"kvf-example.log"};
	auto const& log = kvf::example::log;
	try {
		auto assets_dir = std::string_view{"."};
		auto force_x11 = false;
		auto const build_version = std::format("{}", kvf::build_version_v);
		auto spec = clap::spec::Parameters{
			.parameters =
				{
					clap::named_option(assets_dir, "a,assets", "example assets directory"),
					clap::named_flag(force_x11, "f,force-x11"),
				},
			.program =
				clap::Program{
					.version = build_version,
				},
		};
		auto parser = clap::Parser{std::move(spec)};
		auto const parse_result = parser.parse_main(argc, argv);
		if (parse_result.should_early_exit()) { return parse_result.return_code(); }
		log.info("Using assets directory: {}", assets_dir);
		if (force_x11) { glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11); }
		kvf::example::App{build_version}.run(assets_dir);
	} catch (std::exception const& e) {
		log.error("PANIC: {}", e.what());
		return EXIT_FAILURE;
	} catch (...) {
		log.error("PANIC: Unknown");
		return EXIT_FAILURE;
	}
}
