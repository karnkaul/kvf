#pragma once
#include <klib/unique.hpp>
#include <kvf/bitmap.hpp>

namespace kvf {
class ImageBitmap {
  public:
	static constexpr auto channels_v{Bitmap::channels_v};

	ImageBitmap() = default;

	explicit ImageBitmap(std::span<std::byte const> compressed);

	auto decompress(std::span<std::byte const> compressed) -> bool;

	[[nodiscard]] auto is_loaded() const -> bool { return !m_bitmap.is_identity(); }
	[[nodiscard]] auto bitmap() const -> Bitmap { return m_bitmap.get(); }

  private:
	struct Id {
		constexpr auto operator()(Bitmap const& a) const -> bool { return a.bytes.empty(); }
	};
	struct Deleter {
		void operator()(Bitmap const& bitmap) const noexcept;
	};
	klib::Unique<Bitmap, Deleter, Id> m_bitmap{};
};
} // namespace kvf
