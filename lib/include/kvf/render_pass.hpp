#pragma once
#include <kvf/device_block.hpp>
#include <kvf/image.hpp>
#include <kvf/render_device.hpp>

namespace kvf {
class RenderPass {
  public:
	explicit RenderPass(gsl::not_null<RenderDevice*> render_device, vk::Extent2D extent);

	void set_color_target(vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);
	void set_depth_target();

	void resize(vk::Extent2D extent);

	[[nodiscard]] auto render_target() const -> RenderTarget;

	void begin_render(vk::CommandBuffer command_buffer);
	void end_render();

	vk::ClearColorValue clear_color{};
	vk::ClearDepthStencilValue clear_depth{1.0f, 0};
	vk::AttachmentStoreOp depth_store_op{vk::AttachmentStoreOp::eDontCare};

  protected:
	struct Framebuffer {
		vma::Image color{};
		vma::Image resolve{};
		vma::Image depth{};

		[[nodiscard]] auto has_color_target() const -> bool { return color.get_image() != vk::Image{}; }
		[[nodiscard]] auto has_resolve_target() const -> bool { return resolve.get_image() != vk::Image{}; }
		[[nodiscard]] auto has_depth_target() const -> bool { return depth.get_image() != vk::Image{}; }
	};

	struct Targets {
		RenderTarget color{};
		RenderTarget resolve{};
		RenderTarget depth{};
	};

	void set_render_targets();

	[[nodiscard]] auto color_image_info(vk::SampleCountFlagBits samples) const -> vma::ImageCreateInfo;
	[[nodiscard]] auto depth_image_info() const -> vma::ImageCreateInfo;

	RenderDevice* m_device{};
	vk::SampleCountFlagBits m_samples{};

	Buffered<Framebuffer> m_framebuffers{};

	vk::CommandBuffer m_command_buffer{};
	vk::Extent2D m_extent{};

	Targets m_render_targets{};

  private:
	std::vector<vk::ImageMemoryBarrier2> m_barriers{};

	DeviceBlock m_device_block{};
};
} // namespace kvf
