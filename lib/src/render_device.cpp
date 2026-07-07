#include "kvf/render_device.hpp"
#include "kvf/build_version.hpp"
#include "kvf/device_waiter.hpp"
#include "kvf/panic.hpp"
#include "kvf/ring.hpp"
#include "kvf/util.hpp"
#include "log.hpp"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <glm/gtc/color_space.hpp>
#include <glm/mat4x4.hpp>
#include <imgui.h>
#include <mutex>
#include <optional>
#include <ranges>

namespace kvf {
namespace {
[[nodiscard]] constexpr auto glfw_platform_to_string_view(int const platform) -> std::string_view {
	switch (platform) {
	case GLFW_PLATFORM_NULL: return "Null";
	case GLFW_PLATFORM_WIN32: return "Win32";
	case GLFW_PLATFORM_X11: return "X11";
	case GLFW_PLATFORM_WAYLAND: return "Wayland";
	case GLFW_PLATFORM_COCOA: return "Cocoa";
	default: return "[unknown]";
	}
}

[[nodiscard]] auto instance_extensions() -> std::span<char const* const> {
	auto count = std::uint32_t{};
	auto const* first = glfwGetRequiredInstanceExtensions(&count);
	return {first, count};
}

[[nodiscard]] auto optimal_depth_format(vk::PhysicalDevice const& gpu) -> vk::Format {
	static constexpr auto target_v{vk::Format::eD32Sfloat};
	auto const props = gpu.getFormatProperties(target_v);
	if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) { return target_v; }
	return vk::Format::eD16Unorm;
}

[[nodiscard]] auto filter_present_modes(std::span<vk::PresentModeKHR const> all) -> std::vector<vk::PresentModeKHR> {
	auto ret = std::vector<vk::PresentModeKHR>{};
	for (auto const in : all) {
		if (std::ranges::find(supported_present_modes_v, in) == supported_present_modes_v.end()) { continue; }
		ret.push_back(in);
	}
	return ret;
}

[[nodiscard]] constexpr auto optimal_present_mode(std::span<vk::PresentModeKHR const> present_modes) -> vk::PresentModeKHR {
	constexpr auto desired_v = std::array{vk::PresentModeKHR::eFifoRelaxed, vk::PresentModeKHR::eFifo, vk::PresentModeKHR::eMailbox};
	for (auto const desired : desired_v) {
		if (std::ranges::find(present_modes, desired) != present_modes.end()) { return desired; }
	}
	return vk::PresentModeKHR::eFifo;
}

[[nodiscard]] constexpr auto compatible_surface_format(std::span<vk::SurfaceFormatKHR const> supported, bool const linear)
	-> std::optional<vk::SurfaceFormatKHR> {
	auto const& viable_formats = linear ? util::linear_formats_v : util::srgb_formats_v;
	for (auto const desired_format : viable_formats) {
		auto const it = std::ranges::find_if(supported, [desired_format](vk::SurfaceFormatKHR const& format) {
			return format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear && format.format == desired_format;
		});
		if (it != supported.end()) { return *it; }
	}
	return {};
}

struct LayoutMasks {
	vk::ImageLayout layout{};
	vk::AccessFlags2 access{};
	vk::PipelineStageFlags2 stage{};
};

constexpr auto source_layout_masks_v = std::array{
	LayoutMasks{.layout = vk::ImageLayout::eUndefined, .stage = vk::PipelineStageFlagBits2::eTopOfPipe},
	LayoutMasks{.layout = vk::ImageLayout::ePresentSrcKHR, .stage = vk::PipelineStageFlagBits2::eTopOfPipe},
	LayoutMasks{
		.layout = vk::ImageLayout::eTransferDstOptimal,
		.access = vk::AccessFlagBits2::eTransferWrite,
		.stage = vk::PipelineStageFlagBits2::eTransfer,
	},
	LayoutMasks{
		.layout = vk::ImageLayout::eAttachmentOptimal,
		.access = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
		.stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	},
};

constexpr auto target_layout_masks_v = std::array{
	LayoutMasks{
		.layout = vk::ImageLayout::eTransferDstOptimal,
		.access = vk::AccessFlagBits2::eTransferWrite,
		.stage = vk::PipelineStageFlagBits2::eTransfer,
	},
	LayoutMasks{
		.layout = vk::ImageLayout::eAttachmentOptimal,
		.access = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
		.stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	},
	LayoutMasks{.layout = vk::ImageLayout::ePresentSrcKHR, .stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput},
};

constexpr auto layout_masks_for(vk::ImageLayout const layout, std::span<LayoutMasks const> masks) {
	auto const it = std::ranges::find_if(masks, [layout](LayoutMasks const& lm) { return lm.layout == layout; });
	KLIB_ASSERT(it != masks.end());
	return *it;
}

class VulkanAllocator {
  public:
	VulkanAllocator(VulkanAllocator const&) = delete;
	VulkanAllocator(VulkanAllocator&&) = delete;
	VulkanAllocator& operator=(VulkanAllocator const&) = delete;
	VulkanAllocator& operator=(VulkanAllocator&&) = delete;

	explicit VulkanAllocator(vk::Instance const instance, Gpu const& gpu, vk::Device const device) {
		auto allocator_ci = VmaAllocatorCreateInfo{};
		allocator_ci.instance = instance;
		allocator_ci.physicalDevice = gpu.device;
		allocator_ci.device = device;
		auto dl = VULKAN_HPP_DEFAULT_DISPATCHER;
		auto vkFunc = VmaVulkanFunctions{};
		vkFunc.vkGetInstanceProcAddr = dl.vkGetInstanceProcAddr;
		vkFunc.vkGetDeviceProcAddr = dl.vkGetDeviceProcAddr;
		allocator_ci.pVulkanFunctions = &vkFunc;
		if (vmaCreateAllocator(&allocator_ci, &m_allocator) != VK_SUCCESS) { throw Panic{"Failed to create Vulkan Allocator"}; }
	}

	~VulkanAllocator() { vmaDestroyAllocator(m_allocator); }

	[[nodiscard]] auto get() const -> VmaAllocator { return m_allocator; }

	operator VmaAllocator() const { return get(); }

  private:
	VmaAllocator m_allocator{};
};

struct GpuList {
	std::vector<Gpu> gpus{};
	std::vector<std::uint32_t> queue_families{};

	static constexpr auto has_required_extensions(std::span<vk::ExtensionProperties const> available) {
		auto const has_extension = [available](char const* name) {
			auto const match = [name](vk::ExtensionProperties const& props) { return std::string_view{props.extensionName.data()} == name; };
			return std::ranges::find_if(available, match) != available.end();
		};
		return has_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	}

	[[nodiscard]] static auto get_viable(vk::Instance instance, vk::SurfaceKHR surface) {
		static constexpr auto queue_flags_v = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eTransfer;
		auto const all_devices = instance.enumeratePhysicalDevices();
		auto ret = GpuList{};
		auto const get_queue_family = [](vk::PhysicalDevice device, std::uint32_t& out) {
			for (auto const& [index, family] : std::ranges::enumerate_view(device.getQueueFamilyProperties())) {
				if ((family.queueFlags & queue_flags_v) != queue_flags_v) { continue; }
				out = static_cast<std::uint32_t>(index);
				return true;
			}
			return false;
		};
		for (auto const& device : all_devices) {
			if (device.getProperties().apiVersion < VK_API_VERSION_1_3) { continue; }
			if (!has_required_extensions(device.enumerateDeviceExtensionProperties())) { continue; }
			auto queue_family = std::uint32_t{};
			if (!get_queue_family(device, queue_family)) { continue; }
			if (device.getSurfaceSupportKHR(queue_family, surface) == vk::False) { continue; }
			ret.gpus.push_back(Gpu{.device = device, .properties = device.getProperties(), .features = device.getFeatures()});
			ret.queue_families.push_back(queue_family);
		}
		return ret;
	}

	[[nodiscard]] auto get_queue_family(gsl::not_null<Gpu const*> gpu) const -> std::uint32_t {
		for (auto [g, q] : std::ranges::zip_view(gpus, queue_families)) {
			if (&g == gpu) { return q; }
		}
		throw Panic{"Invalid GPU"};
	}
};

class Swapchain {
  public:
	void initialize(vk::Device device, vk::PhysicalDevice physical_device, vk::SwapchainCreateInfoKHR const& info) {
		m_device = device;
		m_physical_device = physical_device;
		m_info = info;
		m_info.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst).setImageArrayLayers(1);
	}

	void recreate(vk::Extent2D const framebuffer, std::optional<vk::PresentModeKHR> present_mode = {}) {
		if (framebuffer.width == 0 || framebuffer.height == 0) { return; }

		auto const surface_capabilities = m_physical_device.getSurfaceCapabilitiesKHR(m_info.surface);
		m_info.imageExtent = get_image_extent(surface_capabilities, framebuffer);
		if (present_mode) { m_info.presentMode = *present_mode; }
		m_info.minImageCount = get_image_count(surface_capabilities);
		if (m_info.minImageCount <= KVF_RESOURCE_BUFFERING) {
			throw Panic{std::format("Insufficient Swapchain images: {}, KVF_RESOURCE_BUFFERING: {}", m_info.minImageCount, KVF_RESOURCE_BUFFERING)};
		}
		m_info.oldSwapchain = *m_swapchain;

		m_device.waitIdle();
		m_swapchain = m_device.createSwapchainKHRUnique(m_info);
		if (!m_swapchain) { throw Panic{"Failed to create Vulkan Swapchain"}; }

		auto image_count = std::uint32_t{};
		if (m_device.getSwapchainImagesKHR(*m_swapchain, &image_count, nullptr) != vk::Result::eSuccess) { throw Panic{"Failed to get Swapchain Images"}; }
		m_images.resize(image_count);
		if (m_device.getSwapchainImagesKHR(*m_swapchain, &image_count, m_images.data()) != vk::Result::eSuccess) {
			throw Panic{"Failed to get Swapchain Images"};
		}

		m_image_views.clear();
		m_image_views.reserve(m_images.size());
		auto image_view_ci = util::ImageViewCreateInfo{
			.image = vk::Image{},
			.format = m_info.imageFormat,
			.subresource = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
			.type = vk::ImageViewType::e2D,
		};
		for (auto const image : m_images) {
			image_view_ci.image = image;
			m_image_views.push_back(util::create_image_view(m_device, image_view_ci));
		}

		m_present_sems.clear();
		m_present_sems.resize(m_images.size());
		for (auto& semaphore : m_present_sems) { semaphore = m_device.createSemaphoreUnique({}); }

		m_image_index.reset();
		m_layout = vk::ImageLayout::eUndefined;

		auto const extent = m_info.imageExtent;
		std::string_view const color_space = util::is_srgb(m_info.imageFormat) ? "sRGB" : "Linear";
		log.debug("Swapchain color-space: {}, extent: {}x{}, mode: {}", color_space, extent.width, extent.height, util::to_string_view(m_info.presentMode));
	}

	[[nodiscard]] auto get_image_index() const -> std::optional<std::uint32_t> { return m_image_index; }

	auto acquire_next_image(vk::Semaphore const signal) -> bool {
		if (m_image_index) { return true; }

		static constexpr std::chrono::nanoseconds timeout_v = 5s;

		auto image_index = std::uint32_t{};
		auto const result = m_device.acquireNextImageKHR(*m_swapchain, timeout_v.count(), signal, {}, &image_index);
		switch (result) {
		case vk::Result::eErrorOutOfDateKHR: return false;
		case vk::Result::eSuboptimalKHR:
		case vk::Result::eSuccess: m_image_index = image_index; return true;
		default: throw Panic{"Failed to acquire Swapchain Image"};
		}
	}

	auto present(vk::Queue queue) -> bool {
		KLIB_ASSERT(m_image_index);

		auto const semaphore = *m_present_sems.at(*m_image_index);
		auto pi = vk::PresentInfoKHR{};
		pi.setImageIndices(*m_image_index).setSwapchains(*m_swapchain).setWaitSemaphores(semaphore);
		auto const result = queue.presentKHR(&pi);
		m_image_index.reset();

		switch (result) {
		case vk::Result::eErrorOutOfDateKHR: return false;
		case vk::Result::eSuboptimalKHR:
		case vk::Result::eSuccess: return true;
		default: throw Panic{"Failed to present Swapchain Image"};
		}
	}

	[[nodiscard]] auto get_info() const -> vk::SwapchainCreateInfoKHR const& { return m_info; }
	[[nodiscard]] auto get_image() const -> vk::Image { return m_images.at(m_image_index.value()); }
	[[nodiscard]] auto get_image_view() const -> vk::ImageView { return *m_image_views.at(m_image_index.value()); }
	[[nodiscard]] auto get_present_semaphore() const -> vk::Semaphore { return *m_present_sems.at(m_image_index.value()); }

  private:
	static constexpr std::uint32_t min_images_v{KVF_RESOURCE_BUFFERING + 1};

	[[nodiscard]] static constexpr auto get_image_extent(vk::SurfaceCapabilitiesKHR const& caps, vk::Extent2D framebuffer) -> vk::Extent2D {
		constexpr auto limitless_v = std::numeric_limits<std::uint32_t>::max();
		if (caps.currentExtent.width < limitless_v && caps.currentExtent.height < limitless_v) { return caps.currentExtent; }
		auto const x = std::clamp(framebuffer.width, caps.minImageExtent.width, caps.maxImageExtent.width);
		auto const y = std::clamp(framebuffer.height, caps.minImageExtent.height, caps.maxImageExtent.height);
		return vk::Extent2D{x, y};
	}

	[[nodiscard]] static constexpr auto get_image_count(vk::SurfaceCapabilitiesKHR const& caps) -> std::uint32_t {
		if (caps.maxImageCount < caps.minImageCount) { return std::max(min_images_v, caps.minImageCount); }
		return std::clamp(min_images_v, caps.minImageCount, caps.maxImageCount);
	}

	vk::PhysicalDevice m_physical_device{};
	vk::Device m_device{};
	std::vector<vk::PresentModeKHR> m_present_modes{};

	vk::SwapchainCreateInfoKHR m_info{};
	vk::UniqueSwapchainKHR m_swapchain{};
	std::vector<vk::Image> m_images{};
	std::vector<vk::UniqueImageView> m_image_views{};
	std::vector<vk::UniqueSemaphore> m_present_sems{};

	std::optional<std::uint32_t> m_image_index{};
	vk::ImageLayout m_layout{};
};

class DearImGui {
  public:
	struct CreateInfo {
		GLFWwindow* window{};
		std::uint32_t api_version{};
		vk::Instance instance{};
		vk::PhysicalDevice physical_device{};
		std::uint32_t queue_family{};
		vk::Device device{};
		vk::Queue queue{};
		vk::Format color_format{};
		vk::SampleCountFlagBits samples{};
		bool srgb_target{};
	};

	DearImGui(DearImGui const&) = delete;
	DearImGui(DearImGui&&) = delete;
	DearImGui& operator=(DearImGui const&) = delete;
	DearImGui& operator=(DearImGui&&) = delete;

	explicit DearImGui(CreateInfo const& create_info) {
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		static auto const load_vk_func = +[](char const* name, void* user_data) {
			return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(*static_cast<vk::Instance*>(user_data), name);
		};
		auto instance = create_info.instance;
		ImGui_ImplVulkan_LoadFunctions(create_info.api_version, load_vk_func, &instance);

		if (!ImGui_ImplGlfw_InitForVulkan(create_info.window, true)) { throw std::runtime_error{"Failed to initialize Dear ImGui"}; }

		auto init_info = ImGui_ImplVulkan_InitInfo{};
		init_info.ApiVersion = create_info.api_version;
		init_info.Instance = create_info.instance;
		init_info.PhysicalDevice = create_info.physical_device;
		init_info.Device = create_info.device;
		init_info.QueueFamily = create_info.queue_family;
		init_info.Queue = create_info.queue;
		init_info.MinImageCount = 2;
		init_info.ImageCount = static_cast<std::uint32_t>(resource_buffering_v);
		init_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE;
		init_info.UseDynamicRendering = true;
		init_info.PipelineInfoMain.MSAASamples = static_cast<VkSampleCountFlagBits>(create_info.samples);
		auto pipline_rendering_ci = vk::PipelineRenderingCreateInfo{};
		pipline_rendering_ci.setColorAttachmentCount(1).setColorAttachmentFormats(create_info.color_format);
		init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipline_rendering_ci;
		if (!ImGui_ImplVulkan_Init(&init_info)) { throw std::runtime_error{"Failed to initialize Dear ImGui"}; }

		ImGui::StyleColorsDark();
		if (create_info.srgb_target) {
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
			for (auto& colour : ImGui::GetStyle().Colors) {
				auto const linear = glm::convertSRGBToLinear(glm::vec4{colour.x, colour.y, colour.z, colour.w});
				colour = ImVec4{linear.x, linear.y, linear.z, linear.w};
			}
			ImGui::GetStyle().Colors[ImGuiCol_WindowBg].w = 0.99f; // more opaque
		}

		m_device = create_info.device;
	}

	~DearImGui() {
		m_device.waitIdle();
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void new_frame() {
		if (m_state == State::eEndFrame) { end_frame(); }
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		m_state = State::eEndFrame;
	}

	void end_frame() {
		if (m_state == State::eNewFrame) { return; }
		// ImGui::Render calls ImGui::EndFrame
		ImGui::Render();
		m_state = State::eNewFrame;
	}

	// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
	void render(vk::CommandBuffer const command_buffer) const {
		if (auto* data = ImGui::GetDrawData()) { ImGui_ImplVulkan_RenderDrawData(data, command_buffer); }
	}

  private:
	enum class State : std::int8_t { eNewFrame, eEndFrame };

	State m_state{};

	vk::Device m_device{};
};

class RingDescriptorAllocator : public IRingDescriptorAllocator, public INextFrameListener {
  public:
	explicit RingDescriptorAllocator(vk::Device device, std::span<vk::DescriptorPoolSize const> pool_sizes, std::uint32_t const sets_per_pool) {
		for (auto& allocator : m_pools) { allocator.initialize(device, pool_sizes, sets_per_pool); }
	}

  private:
	class Pool {
	  public:
		void initialize(vk::Device device, std::span<vk::DescriptorPoolSize const> pool_sizes, std::uint32_t const sets_per_pool) {
			m_device = device;
			m_pool_sizes = pool_sizes;
			m_sets_per_pool = sets_per_pool;
		}

		void reset_pools() {
			for (auto& pool : m_pools) { m_device.resetDescriptorPool(*pool); }
			m_index = 0;
		}

		auto allocate_next(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> set_layouts) -> bool {
			KLIB_ASSERT(m_device && !m_pool_sizes.empty() && m_sets_per_pool > 0);
			if (set_layouts.empty() || out_sets.size() != set_layouts.size()) { return false; }
			auto result = try_allocate(out_sets, set_layouts);
			switch (result) {
			case vk::Result::eSuccess: return true;
			case vk::Result::eErrorOutOfPoolMemory: ++m_index; return try_allocate(out_sets, set_layouts) == vk::Result::eSuccess;
			default: return false;
			}
		}

	  private:
		auto get_pool() -> vk::DescriptorPool {
			if (m_index > 4096) { log.warn("{} DescriptorPools allocated this frame", m_index); }
			if (m_index >= m_pools.size()) {
				m_index = m_pools.size();
				auto dpci = vk::DescriptorPoolCreateInfo{};
				dpci.setPoolSizes(m_pool_sizes).setMaxSets(m_sets_per_pool);
				m_pools.push_back(m_device.createDescriptorPoolUnique(dpci));
			}
			return *m_pools.at(m_index);
		}

		auto try_allocate(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> vk::Result {
			auto const pool = get_pool();
			auto dsai = vk::DescriptorSetAllocateInfo{};
			dsai.setDescriptorPool(pool).setSetLayouts(layouts);
			return m_device.allocateDescriptorSets(&dsai, out_sets.data());
		}

		vk::Device m_device{};
		std::span<vk::DescriptorPoolSize const> m_pool_sizes{};
		std::uint32_t m_sets_per_pool{};

		std::vector<vk::UniqueDescriptorPool> m_pools{};
		std::size_t m_index{};
	};

	auto allocate_next(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> set_layouts) -> bool final {
		return m_pools.at(std::size_t(m_frame_index)).allocate_next(out_sets, set_layouts);
	}

	void on_next_frame(FrameIndex const frame_index) final {
		m_frame_index = frame_index;
		m_pools.at(std::size_t(m_frame_index)).reset_pools();
	}

	Ring<Pool> m_pools{};
	FrameIndex m_frame_index{};
};

class RenderDevice : public IRenderDevice {
  public:
	RenderDevice(gsl::not_null<GLFWwindow*> window, CreateInfo const& create_info) : m_window(window), m_flags(create_info.flags) {
		log.debug("kvf {}, platform: {}", build_version_v, glfw_platform_to_string_view(glfwGetPlatform()));
		create_instance();
		create_surface();
		select_gpu(create_info.gpu_selector);
		create_device();
		create_swapchain();
		m_allocator.emplace(*m_instance, m_gpu, *m_device);

		create_dear_imgui();

		create_syncs();
		create_command_buffers();

		create_descriptor_allocator(create_info.custom_pool_sizes, create_info.sets_per_pool);

		m_dear_imgui->new_frame();
	}

  private:
	struct Sync {
		vk::UniqueSemaphore draw{};
		vk::UniqueFence drawn{};
	};

	struct Deleter {
		void operator()(VmaAllocator allocator) const noexcept { vmaDestroyAllocator(allocator); }
	};

	[[nodiscard]] auto get_window() const -> gsl::not_null<GLFWwindow*> final { return m_window; }
	[[nodiscard]] auto get_instance() const -> vk::Instance final { return *m_instance; }
	[[nodiscard]] auto get_gpu() const -> Gpu const& final { return m_gpu; }
	[[nodiscard]] auto get_surface() const -> vk::SurfaceKHR final { return *m_surface; }
	[[nodiscard]] auto get_device() const -> vk::Device final { return *m_device; }
	[[nodiscard]] auto get_queue_family() const -> std::uint32_t final { return m_queue_family; }
	[[nodiscard]] auto get_allocator() const -> VmaAllocator final { return *m_allocator; }

	[[nodiscard]] auto get_swapchain_image_extent() const -> vk::Extent2D final { return m_swapchain.get_info().imageExtent; }
	[[nodiscard]] auto get_swapchain_color_format() const -> vk::Format final { return m_swapchain.get_info().imageFormat; }
	[[nodiscard]] auto get_optimal_depth_format() const -> vk::Format final { return m_optimal_depth_format; }

	[[nodiscard]] auto get_loader_api_version() const -> klib::Version final { return m_loader_version; }
	[[nodiscard]] auto get_flags() const -> RenderDeviceFlag final { return m_flags; }

	[[nodiscard]] auto get_present_mode() const -> vk::PresentModeKHR final { return m_swapchain.get_info().presentMode; }
	[[nodiscard]] auto get_supported_present_modes() const -> std::span<vk::PresentModeKHR const> final { return m_present_modes; }

	void set_present_mode(vk::PresentModeKHR present_mode) final {
		if (std::ranges::find(m_present_modes, present_mode) == m_present_modes.end()) { return; }
		auto const extent = util::to_vk_extent(util::framebuffer_size(m_window.get()));
		m_swapchain.recreate(extent, present_mode);
	}

	[[nodiscard]] auto get_render_imgui() const -> bool final { return m_render_imgui; }
	void set_render_imgui(bool should_render) final { m_render_imgui = should_render; }

	[[nodiscard]] auto get_frame_index() const -> FrameIndex final { return FrameIndex{m_frame_index}; }
	void attach_next_frame_listener(std::weak_ptr<INextFrameListener> listener) final { m_next_frame_listeners.push_back(std::move(listener)); }

	[[nodiscard]] auto get_descriptor_allocator() -> IRingDescriptorAllocator& final { return *m_descriptor_allocator; }

	void queue_submit(vk::SubmitInfo2 const& si, vk::Fence const fence) final {
		auto const lock = std::scoped_lock{m_mutex};
		m_queue.submit2(si, fence);
	}

	auto next_frame() -> vk::CommandBuffer final {
		begin_frame();
		return m_current_cmd;
	}

	auto render(RenderTarget const& render_target, vk::Filter const filter) -> bool final {
		auto const ret = acquire_next_image();
		if (ret) {
			perform_render(render_target, filter);
		} else if (m_current_cmd) {
			m_current_cmd.end();
		}
		m_current_cmd = vk::CommandBuffer{};
		return ret;
	}

	[[nodiscard]] auto get_framebuffer_extent() const -> vk::Extent2D { return util::to_vk_extent(util::framebuffer_size(m_window.get())); }

	void create_instance() {
		VULKAN_HPP_DEFAULT_DISPATCHER.init();

		auto const vk_api_version = vk::enumerateInstanceVersion();
		m_loader_version = klib::Version{
			.major = VK_VERSION_MAJOR(vk_api_version),
			.minor = VK_VERSION_MINOR(vk_api_version),
			.patch = VK_VERSION_PATCH(vk_api_version),
		};
		log.debug("Vulkan loader (Instance API) version: {}", m_loader_version);

		auto app_info = vk::ApplicationInfo{};
		app_info.setApiVersion(VK_MAKE_VERSION(vk_api_version_v.major, vk_api_version_v.minor, vk_api_version_v.patch));

		auto ici = vk::InstanceCreateInfo{};
		auto const wsi_extensions = instance_extensions();
		auto layers = std::vector<char const*>{};
		if ((m_flags & RenderDeviceFlag::ShaderObjectLayer) == RenderDeviceFlag::ShaderObjectLayer) { layers.push_back("VK_LAYER_KHRONOS_shader_object"); }
		ici.setPApplicationInfo(&app_info).setPEnabledExtensionNames(wsi_extensions).setPEnabledLayerNames(layers);
		m_instance = vk::createInstanceUnique(ici);
		if (!m_instance) { throw Panic{"Failed to create Vulkan Instance"}; }

		VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_instance);
		log.debug("Vulkan {} Instance created", vk_api_version_v);
	}

	void create_surface() {
		VkSurfaceKHR surface{};
		glfwCreateWindowSurface(*m_instance, m_window, nullptr, &surface);
		if (surface == VK_NULL_HANDLE) { throw Panic{"Failed to create Vulkan Surface"}; }
		m_surface = vk::UniqueSurfaceKHR{surface, *m_instance};
	}

	void select_gpu(klib::Ptr<Gpu::Selector const> p_selector) {
		static auto const default_gpu_selector = Gpu::Selector{};
		auto const& selector = p_selector ? *p_selector : default_gpu_selector;
		auto const devices = m_instance->enumeratePhysicalDevices();
		auto list = GpuList::get_viable(*m_instance, *m_surface);
		if (list.gpus.empty()) { throw Panic{"No viable GPUs"}; }
		auto const gpu = selector.select(list.gpus);
		m_queue_family = list.get_queue_family(gpu);
		m_gpu = *gpu;
		m_optimal_depth_format = optimal_depth_format(m_gpu.device);
		log.debug("Using GPU: {}, queue family: {}", m_gpu.properties.deviceName.data(), m_queue_family);
	}

	void create_device() {
		auto qci = vk::DeviceQueueCreateInfo{};
		static constexpr auto queue_priorities_v = std::array{1.0f};
		qci.setQueueFamilyIndex(m_queue_family).setQueueCount(1).setQueuePriorities(queue_priorities_v);

		auto enabled_features = vk::PhysicalDeviceFeatures{};
		enabled_features.fillModeNonSolid = m_gpu.features.fillModeNonSolid;
		enabled_features.wideLines = m_gpu.features.wideLines;
		enabled_features.samplerAnisotropy = m_gpu.features.samplerAnisotropy;
		enabled_features.sampleRateShading = m_gpu.features.sampleRateShading;

		auto dr_feature = vk::PhysicalDeviceDynamicRenderingFeatures{vk::True};
		auto sync_feature = vk::PhysicalDeviceSynchronization2Features{vk::True, &dr_feature};
		auto shader_obj_feature = vk::PhysicalDeviceShaderObjectFeaturesEXT{vk::True};

		auto dci = vk::DeviceCreateInfo{};
		auto extensions = std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
		if ((m_flags & RenderDeviceFlag::ShaderObjectFeature) == RenderDeviceFlag::ShaderObjectFeature) {
			dr_feature.setPNext(&shader_obj_feature);
			extensions.push_back("VK_EXT_shader_object");
		}
		dci.setPEnabledExtensionNames(extensions).setQueueCreateInfos(qci).setPEnabledFeatures(&enabled_features).setPNext(&sync_feature);

		m_device = m_gpu.device.createDeviceUnique(dci);
		if (!m_device) { throw Panic{"Failed to create Vulkan Device"}; }
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_device);

		m_queue = m_device->getQueue(m_queue_family, 0);
		log.debug("Vulkan Device created");

		m_device_waiter.get() = *m_device;
	}

	void create_swapchain() {
		auto const is_linear_backbuffer = (m_flags & RenderDeviceFlag::LinearBackbuffer) == RenderDeviceFlag::LinearBackbuffer;
		auto const surface_format = compatible_surface_format(m_gpu.device.getSurfaceFormatsKHR(*m_surface), is_linear_backbuffer);
		if (!surface_format) { throw Panic{"Failed to find compatible Vulkan Surface Format"}; }
		m_present_modes = filter_present_modes(m_gpu.device.getSurfacePresentModesKHR(*m_surface));
		auto sci = vk::SwapchainCreateInfoKHR{};
		sci.setSurface(*m_surface)
			.setPresentMode(optimal_present_mode(m_present_modes))
			.setImageFormat(surface_format->format)
			.setImageColorSpace(surface_format->colorSpace)
			.setQueueFamilyIndices(m_queue_family);
		m_swapchain.initialize(*m_device, m_gpu.device, sci);
		m_swapchain.recreate(get_framebuffer_extent());
	}

	void create_command_buffers() {
		auto command_pool_ci = vk::CommandPoolCreateInfo{};
		command_pool_ci.setQueueFamilyIndex(m_queue_family).setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
		m_command_pool = m_device->createCommandPoolUnique(command_pool_ci);
		auto command_buffer_ai = vk::CommandBufferAllocateInfo{};
		command_buffer_ai.setCommandPool(*m_command_pool)
			.setLevel(vk::CommandBufferLevel::ePrimary)
			.setCommandBufferCount(std::uint32_t(m_command_buffers.size()));
		if (m_device->allocateCommandBuffers(&command_buffer_ai, m_command_buffers.data()) != vk::Result::eSuccess) {
			throw Panic{"Failed to allocate render CommandBuffer(s)"};
		}
	}

	void create_syncs() {
		for (auto& sync : m_syncs) {
			sync.draw = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
			sync.drawn = m_device->createFenceUnique(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
		}
	}

	void create_dear_imgui() {
		auto const is_linear_backbuffer = (m_flags & RenderDeviceFlag::LinearBackbuffer) == RenderDeviceFlag::LinearBackbuffer;
		auto const dear_imgui_ci = DearImGui::CreateInfo{
			.window = m_window,
			.instance = *m_instance,
			.physical_device = m_gpu.device,
			.queue_family = m_queue_family,
			.device = *m_device,
			.queue = m_queue,
			.color_format = m_swapchain.get_info().imageFormat,
			.samples = vk::SampleCountFlagBits::e1,
			.srgb_target = !is_linear_backbuffer,
		};
		m_dear_imgui.emplace(dear_imgui_ci);
		log.debug("Dear ImGui initialized");
	}

	void create_descriptor_allocator(std::span<vk::DescriptorPoolSize const> pool_sizes, std::uint32_t sets_per_pool) {
		if (pool_sizes.empty()) {
			static constexpr auto descriptors_per_type_v{8};
			static constexpr auto pool_sizes_v = std::array{
				vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, descriptors_per_type_v},
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, descriptors_per_type_v},
				vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, descriptors_per_type_v},
			};
			pool_sizes = pool_sizes_v;
		}
		if (sets_per_pool == 0) { sets_per_pool = CreateInfo::sets_per_pool_v; }
		m_descriptor_allocator = std::make_shared<RingDescriptorAllocator>(*m_device, pool_sizes, sets_per_pool);
		attach_next_frame_listener(m_descriptor_allocator);
	}

	void begin_frame() {
		auto const drawn = *m_syncs.at(m_frame_index).drawn;
		if (!util::wait_for_fence(*m_device, drawn)) { throw Panic{"Failed to wait for Render Fence"}; }

		glfwPollEvents();
		m_dear_imgui->new_frame();
		std::erase_if(m_next_frame_listeners, [this](std::weak_ptr<INextFrameListener> const& ptr) {
			if (auto listener = ptr.lock()) {
				listener->on_next_frame(FrameIndex{m_frame_index});
				return false;
			}
			return true;
		});

		m_current_cmd = m_command_buffers.at(m_frame_index);
		m_current_cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	}

	auto acquire_next_image() -> bool {
		m_dear_imgui->end_frame();
		if (!m_current_cmd) { return false; }

		auto const framebuffer_extent = get_framebuffer_extent();
		if (framebuffer_extent.width == 0 || framebuffer_extent.height == 0) { return false; }

		if (m_swapchain.get_info().imageExtent != framebuffer_extent) { m_swapchain.recreate(framebuffer_extent); }

		auto const& sync = m_syncs.at(m_frame_index);
		if (!util::wait_for_fence(*m_device, *sync.drawn)) { throw Panic{"Failed to wait for Render Fence"}; }

		auto const out_of_date = [&] {
			auto lock = std::scoped_lock{m_mutex};
			return !m_swapchain.acquire_next_image(*sync.draw);
		}();

		if (out_of_date) {
			m_swapchain.recreate(framebuffer_extent);
			return false;
		}

		m_device->resetFences(*sync.drawn); // must submit after reset

		return true;
	}

	void perform_render(RenderTarget const& frame, vk::Filter const filter) {
		m_backbuffer_layout = vk::ImageLayout::eUndefined;
		auto const backbuffer = RenderTarget{
			.image = m_swapchain.get_image(),
			.view = m_swapchain.get_image_view(),
			.extent = m_swapchain.get_info().imageExtent,
		};
		KLIB_ASSERT(backbuffer.image && backbuffer.view);

		auto barrier = vk::ImageMemoryBarrier2{};
		auto backbuffer_load_op = vk::AttachmentLoadOp::eClear;

		if (frame.image && frame.view) {
			barrier = transition_backbuffer(backbuffer.image, vk::ImageLayout::eTransferDstOptimal);
			util::record_barrier(m_current_cmd, barrier);
			blit_to_backbuffer(frame, backbuffer, m_current_cmd, filter);
			backbuffer_load_op = vk::AttachmentLoadOp::eLoad;
		}

		if (m_render_imgui) {
			barrier = transition_backbuffer(backbuffer.image, vk::ImageLayout::eAttachmentOptimal);
			util::record_barrier(m_current_cmd, barrier);
			auto color_ai = vk::RenderingAttachmentInfo{};
			color_ai.setImageView(backbuffer.view)
				.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
				.setLoadOp(backbuffer_load_op)
				.setStoreOp(vk::AttachmentStoreOp::eStore);
			render_imgui(color_ai, backbuffer.extent);
		}

		barrier = transition_backbuffer(backbuffer.image, vk::ImageLayout::ePresentSrcKHR);
		util::record_barrier(m_current_cmd, barrier);

		m_current_cmd.end();

		auto const& sync = m_syncs.at(m_frame_index);

		auto const present_sempahore = m_swapchain.get_present_semaphore();
		auto const cbsi = vk::CommandBufferSubmitInfo{m_current_cmd};
		auto const wssi = vk::SemaphoreSubmitInfo{*sync.draw, 0, vk::PipelineStageFlagBits2::eTopOfPipe};
		auto const sssi = vk::SemaphoreSubmitInfo{present_sempahore, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput};
		auto si = vk::SubmitInfo2{};
		si.setCommandBufferInfos(cbsi).setWaitSemaphoreInfos(wssi).setSignalSemaphoreInfos(sssi);

		auto lock = std::unique_lock{m_mutex};
		m_queue.submit2(si, *sync.drawn);
		auto const present_sucess = m_swapchain.present(m_queue);
		lock.unlock();

		if (!present_sucess) { m_swapchain.recreate(get_framebuffer_extent()); }

		m_frame_index = (m_frame_index + 1) % resource_buffering_v;
	}

	void render_imgui(vk::RenderingAttachmentInfo const& backbuffer, vk::Extent2D const extent) {
		auto render_area = vk::Rect2D{};
		render_area.setExtent(extent);

		auto ri = vk::RenderingInfo{};
		ri.setColorAttachments(backbuffer).setLayerCount(1).setRenderArea(render_area);
		m_current_cmd.beginRendering(ri);
		m_dear_imgui->render(m_current_cmd);
		m_current_cmd.endRendering();
	}

	void blit_to_backbuffer(RenderTarget const& frame, RenderTarget const& backbuffer, vk::CommandBuffer cmd, vk::Filter filter) const {
		static auto const isr_v = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		auto barrier = create_image_barrier();
		barrier.setImage(frame.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
			.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
			.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
			.setOldLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
			.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setSubresourceRange(isr_v);
		util::record_barrier(cmd, barrier);

		auto const src_offset = vk::Offset3D{std::int32_t(frame.extent.width), std::int32_t(frame.extent.height), 1};
		auto const dst_offset = vk::Offset3D{std::int32_t(backbuffer.extent.width), std::int32_t(backbuffer.extent.height), 1};
		auto ib = vk::ImageBlit2{};
		ib.setSrcOffsets(std::array{vk::Offset3D{}, src_offset})
			.setDstOffsets(std::array{vk::Offset3D{}, dst_offset})
			.setSrcSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
			.setDstSubresource(ib.srcSubresource);
		auto bii = vk::BlitImageInfo2{};
		bii.setSrcImage(frame.image)
			.setDstImage(backbuffer.image)
			.setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
			.setFilter(filter)
			.setRegions(ib);
		cmd.blitImage2(bii);

		barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setOldLayout(barrier.newLayout)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		util::record_barrier(cmd, barrier);
	}

	[[nodiscard]] auto transition_backbuffer(vk::Image backbuffer, vk::ImageLayout const target) -> vk::ImageMemoryBarrier2 {
		static constexpr auto subresource_range_v = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		auto ret = create_image_barrier();
		ret.setImage(backbuffer).setSubresourceRange(subresource_range_v).setOldLayout(m_backbuffer_layout).setNewLayout(target);

		auto masks = layout_masks_for(m_backbuffer_layout, source_layout_masks_v);
		ret.setSrcAccessMask(masks.access).setSrcStageMask(masks.stage);

		masks = layout_masks_for(target, target_layout_masks_v);
		ret.setDstAccessMask(masks.access).setDstStageMask(masks.stage);

		m_backbuffer_layout = target;
		return ret;
	}

	gsl::not_null<GLFWwindow*> m_window;
	RenderDeviceFlag m_flags{};

	klib::Version m_loader_version{};
	vk::UniqueInstance m_instance{};
	Gpu m_gpu{};
	std::uint32_t m_queue_family{};
	vk::Format m_optimal_depth_format{};
	vk::UniqueSurfaceKHR m_surface{};

	vk::UniqueDevice m_device{};

	std::vector<vk::PresentModeKHR> m_present_modes{};
	Swapchain m_swapchain{};
	vk::Queue m_queue{};

	std::optional<VulkanAllocator> m_allocator{};

	std::optional<DearImGui> m_dear_imgui{};

	Ring<Sync> m_syncs{};
	vk::UniqueCommandPool m_command_pool{};
	Ring<vk::CommandBuffer> m_command_buffers{};

	std::shared_ptr<RingDescriptorAllocator> m_descriptor_allocator{};

	std::vector<std::weak_ptr<INextFrameListener>> m_next_frame_listeners{};
	std::size_t m_frame_index{};
	vk::CommandBuffer m_current_cmd{};
	vk::ImageLayout m_backbuffer_layout{};

	bool m_render_imgui{true};

	std::mutex m_mutex{};
	DeviceWaiter m_device_waiter{};
};
} // namespace

auto IRenderDevice::create(gsl::not_null<GLFWwindow*> window, CreateInfo const& create_info) -> std::unique_ptr<IRenderDevice> {
	return std::make_unique<RenderDevice>(window, create_info);
}

auto IRenderDevice::create_sampler(vk::SamplerCreateInfo create_info) const -> vk::UniqueSampler {
	auto const aniso = std::min(create_info.maxAnisotropy, get_gpu().properties.limits.maxSamplerAnisotropy);
	create_info.setAnisotropyEnable(aniso > 0.0f ? vk::True : vk::False).setMaxAnisotropy(aniso);
	return get_device().createSamplerUnique(create_info);
}

auto IRenderDevice::create_shader_objects(ShaderObjectCreateInfo const& create_info) const -> std::array<vk::UniqueShaderEXT, 2> {
	if ((get_flags() & RenderDeviceFlag::ShaderObjectFeature) != RenderDeviceFlag::ShaderObjectFeature) {
		log.warn("Attempt to create ShaderEXT objects without ShaderObjectFeature");
		return {};
	}

	auto const create_shader_ci = [&create_info](std::span<std::uint32_t const> spirv) {
		auto ret = vk::ShaderCreateInfoEXT{};
		ret.setCodeSize(spirv.size_bytes())
			.setPCode(spirv.data())
			.setSetLayouts(create_info.set_layouts)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setPName("main");
		ret.flags |= vk::ShaderCreateFlagBitsEXT::eLinkStage;
		return ret;
	};

	auto shader_cis = std::array{
		create_shader_ci(create_info.vertex_spir_v),
		create_shader_ci(create_info.fragment_spir_v),
	};
	shader_cis[0].setStage(vk::ShaderStageFlagBits::eVertex).setNextStage(vk::ShaderStageFlagBits::eFragment);
	shader_cis[1].setStage(vk::ShaderStageFlagBits::eFragment);

	auto shaders = std::array<vk::ShaderEXT, 2>{};
	auto result = get_device().createShadersEXT(std::uint32_t(shader_cis.size()), shader_cis.data(), nullptr, shaders.data());
	if (result != vk::Result::eSuccess) {
		log.error("Failed to create ShaderEXT objects");
		return {};
	}

	auto ret = std::array<vk::UniqueShaderEXT, 2>{};
	for (auto [in, out] : std::views::zip(shaders, ret)) { out = vk::UniqueShaderEXT{in, get_device()}; }
	return ret;
}

auto IRenderDevice::create_image_barrier(vk::ImageAspectFlags const aspect) const -> vk::ImageMemoryBarrier2 {
	auto ret = vk::ImageMemoryBarrier2{};
	auto const queue_family = get_queue_family();
	ret.setSrcQueueFamilyIndex(queue_family).setDstQueueFamilyIndex(queue_family).subresourceRange.setAspectMask(aspect).setLevelCount(1).setLayerCount(1);
	return ret;
}

auto IRenderDevice::create_pipeline(vk::PipelineLayout layout, PipelineState const& state, PipelineFormat const& format) const -> vk::UniquePipeline {
	auto shader_stages = std::array<vk::PipelineShaderStageCreateInfo, 2>{};
	shader_stages[0].setStage(vk::ShaderStageFlagBits::eVertex).setPName("main").setModule(state.vertex_shader);
	shader_stages[1].setStage(vk::ShaderStageFlagBits::eFragment).setPName("main").setModule(state.fragment_shader);

	auto vertex_input_ci = vk::PipelineVertexInputStateCreateInfo{};
	vertex_input_ci.setVertexAttributeDescriptions(state.vertex_attributes).setVertexBindingDescriptions(state.vertex_bindings);

	auto rasterization_state_ci = vk::PipelineRasterizationStateCreateInfo{};
	rasterization_state_ci.setPolygonMode(state.polygon_mode).setCullMode(state.cull_mode);

	auto depth_stencil_state_ci = vk::PipelineDepthStencilStateCreateInfo{};
	auto const depth_test = (state.flags & PipelineFlag::DepthTest) == PipelineFlag::DepthTest;
	auto const depth_write = (state.flags & PipelineFlag::DepthWrite) == PipelineFlag::DepthWrite;
	depth_stencil_state_ci.setDepthTestEnable(depth_test ? vk::True : vk::False)
		.setDepthCompareOp(state.depth_compare)
		.setDepthWriteEnable(depth_write ? vk::True : vk::False);

	auto const input_assembly_state_ci = vk::PipelineInputAssemblyStateCreateInfo{{}, state.topology};

	auto color_blend_state_ci = vk::PipelineColorBlendStateCreateInfo{};
	color_blend_state_ci.setAttachments(state.blend_state);

	auto const pdscis = std::array{
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
		vk::DynamicState::eLineWidth,
	};
	auto dynamic_state_ci = vk::PipelineDynamicStateCreateInfo{};
	dynamic_state_ci.setDynamicStates(pdscis);

	auto const viewport_state_ci = vk::PipelineViewportStateCreateInfo({}, 1, {}, 1);

	auto multisample_state_ci = vk::PipelineMultisampleStateCreateInfo{};
	multisample_state_ci.setRasterizationSamples(format.samples).setSampleShadingEnable(vk::False);

	auto rendering_ci = vk::PipelineRenderingCreateInfo{};
	if (format.color != vk::Format::eUndefined) { rendering_ci.setColorAttachmentFormats(format.color); }
	rendering_ci.setDepthAttachmentFormat(format.depth);

	auto graphics_pipeline_ci = vk::GraphicsPipelineCreateInfo{};
	graphics_pipeline_ci.setPVertexInputState(&vertex_input_ci)
		.setStages(shader_stages)
		.setPRasterizationState(&rasterization_state_ci)
		.setPDepthStencilState(&depth_stencil_state_ci)
		.setPInputAssemblyState(&input_assembly_state_ci)
		.setPColorBlendState(&color_blend_state_ci)
		.setPDynamicState(&dynamic_state_ci)
		.setPViewportState(&viewport_state_ci)
		.setPMultisampleState(&multisample_state_ci)
		.setLayout(layout)
		.setPNext(&rendering_ci);

	auto const device = get_device();
	auto ret = vk::Pipeline{};
	if (device.createGraphicsPipelines({}, 1, &graphics_pipeline_ci, {}, &ret) != vk::Result::eSuccess) {
		log.error("Failed to create Vulkan Graphics Pipeline");
		return {};
	}

	return vk::UniquePipeline{ret, device};
}
} // namespace kvf
