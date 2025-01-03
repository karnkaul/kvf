#pragma once
#include <concepts>
#include <cstddef>
#include <span>

namespace kvf {
template <typename Type>
concept TrivialT = std::is_trivially_copyable_v<Type>;

class BufferWrite {
  public:
	BufferWrite() = default;

	constexpr BufferWrite(void const* data, std::size_t const size) : m_data(data), m_size(size) {}

	template <TrivialT Type>
	constexpr BufferWrite(Type const& t) : BufferWrite(&t, sizeof(Type)) {}

	template <TrivialT Type, std::size_t Extent>
	constexpr BufferWrite(std::span<Type, Extent> const t) : BufferWrite(t.data(), t.size_bytes()) {}

	template <std::convertible_to<std::span<std::byte const>> Type>
	constexpr BufferWrite(Type const& bytes) : BufferWrite(std::span<std::byte const>{bytes}) {}

	[[nodiscard]] constexpr auto data() const -> void const* { return m_data; }
	[[nodiscard]] constexpr auto size() const -> std::size_t { return m_size; }

	[[nodiscard]] constexpr auto is_empty() const -> bool { return size() == 0; }

  private:
	void const* m_data{};
	std::size_t m_size{};
};
} // namespace kvf
