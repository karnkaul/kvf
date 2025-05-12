#include <app.hpp>
#include <klib/args/parse.hpp>
#include <klib/version_str.hpp>
#include <kvf/build_version.hpp>
#include <log.hpp>

auto main(int argc, char** argv) -> int {
	auto const log_file = klib::log::File{"kvf-example.log"};
	auto const& log = kvf::example::log;
	try {
		auto assets_dir = std::string_view{"."};
		auto const args = std::array{
			klib::args::named_option(assets_dir, "a,assets", "example assets directory"),
		};
		auto const build_version = std::format("{}", kvf::build_version_v);
		auto const parse_info = klib::args::ParseInfo{.version = build_version};
		auto const parse_result = klib::args::parse_main(parse_info, args, argc, argv);
		if (parse_result.early_return()) { return parse_result.get_return_code(); }
		log.info("Using assets directory: {}", assets_dir);
		kvf::example::App{build_version}.run(assets_dir);
	} catch (std::exception const& e) {
		log.error("PANIC: {}", e.what());
		return EXIT_FAILURE;
	} catch (...) {
		log.error("PANIC: Unknown");
		return EXIT_FAILURE;
	}
}
