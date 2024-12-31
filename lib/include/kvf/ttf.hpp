#pragma once
#include <glm/vec2.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace kvf::ttf {
enum struct Codepoint : std::uint32_t {
	eTofu = 0,
	eSpace = 32,
	eAsciiFirst = eSpace,
	eAsciiLast = 126,
};

struct Slot {
	glm::ivec2 size{};
	glm::ivec2 left_top{};
	glm::ivec2 advance{};
	std::span<std::byte const> alpha_channels{};

	[[nodiscard]] constexpr auto operator[](int const x, int const y) const -> std::byte {
		auto const index = std::size_t(y * size.x + x);
		if (index >= alpha_channels.size()) { return {}; }
		return alpha_channels[index];
	}
};

class Typeface {
  public:
	Typeface();

	explicit Typeface(std::vector<std::byte> font) { load(std::move(font)); }

	auto load(std::vector<std::byte> font) -> bool;
	[[nodiscard]] auto is_loaded() const -> bool;

	auto set_height(std::uint32_t height) -> bool;
	auto load_slot(Slot& out, Codepoint codepoint) -> bool;

	explicit operator bool() const { return is_loaded(); }

  private:
	struct Impl;
	struct Deleter {
		void operator()(Impl* ptr) const noexcept;
	};
	std::unique_ptr<Impl, Deleter> m_impl{};
};
} // namespace kvf::ttf
