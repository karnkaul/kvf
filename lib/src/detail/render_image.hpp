#pragma once
#include "detail/vma.hpp"
#include "kvf/kvf_fwd.hpp"
#include "kvf/render_image.hpp"

namespace kvf::detail {
class RenderImage : public IRenderImage {
  public:
	explicit RenderImage(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info);

  private:
	void recreate(CreateInfo const& create_info) final { recreate_impl(create_info); }

	[[nodiscard]] auto get_render_device() const -> IRenderDevice& final { return *m_render_device; }

	[[nodiscard]] auto get_format() const -> vk::Format final { return m_info.format; }
	[[nodiscard]] auto get_aspect() const -> vk::ImageAspectFlags final { return m_info.aspect; }
	[[nodiscard]] auto get_usage() const -> vk::ImageUsageFlags final { return m_info.usage; }
	[[nodiscard]] auto get_samples() const -> vk::SampleCountFlagBits final { return m_info.samples; }
	[[nodiscard]] auto get_layers() const -> std::uint32_t final { return m_info.layers; }
	[[nodiscard]] auto get_view_type() const -> vk::ImageViewType final { return m_info.view_type; }
	[[nodiscard]] auto get_extent() const -> vk::Extent2D final { return m_info.extent; }
	[[nodiscard]] auto get_layout() const -> vk::ImageLayout final { return m_layout; }
	[[nodiscard]] auto get_image_view() const -> vk::ImageView final { return *m_image_view; }
	[[nodiscard]] auto get_image() const -> vk::Image final { return m_image.get().image; }
	[[nodiscard]] auto get_mip_levels() const -> std::uint32_t final { return m_image.get().mip_levels; }

	void resize(vk::Extent2D extent) final;
	auto resize_and_overwrite(std::span<Bitmap const> layers) -> bool final;
	[[nodiscard]] auto copy_to_bitmap(vk::Extent2D custom_extent) const -> ColorBitmap final;

	void transition(vk::CommandBuffer command_buffer, vk::ImageMemoryBarrier2 barrier) final;

	void recreate_impl(CreateInfo create_info);

	gsl::not_null<IRenderDevice*> m_render_device;

	CreateInfo m_info{};
	vma::UniqueImage m_image{};
	vk::UniqueImageView m_image_view{};

	vk::ImageLayout m_layout{};
};
} // namespace kvf::detail
