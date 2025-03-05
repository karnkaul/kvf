#include <vk_mem_alloc.h>
#include <glm/gtc/color_space.hpp>
#include <glm/mat4x4.hpp>
#include <klib/assert.hpp>
#include <klib/debug_trap.hpp>
#include <klib/flex_array.hpp>
#include <klib/scoped_defer.hpp>
#include <klib/unique.hpp>
#include <klib/version_str.hpp>
#include <kvf/build_version.hpp>
#include <kvf/error.hpp>
#include <kvf/is_positive.hpp>
#include <log.hpp>
#include <vulkan/vulkan.hpp>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <ranges>

// common

namespace kvf {
namespace {
template <typename... T>
constexpr void ensure_positive(T&... out) {
	auto const sanitize = [](auto& t) {
		if (!is_positive(t)) { t = 1; }
	};
	(sanitize(out), ...);
}

[[maybe_unused]] void full_barrier(vk::ImageMemoryBarrier2& out) {
	out.setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
		.setDstAccessMask(out.srcAccessMask)
		.setDstStageMask(out.dstStageMask);
}
} // namespace
} // namespace kvf

// window

#include <kvf/window.hpp>

namespace {
void glfw_init() {
	if (glfwInit() != GLFW_TRUE) { throw kvf::Error{"Failed to initialize GLFW"}; }
	if (glfwVulkanSupported() != GLFW_TRUE) { throw kvf::Error{"Vulkan not supported"}; }
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
}
} // namespace

void kvf::WindowDeleter::operator()(GLFWwindow* ptr) const noexcept {
	glfwDestroyWindow(ptr);
	glfwTerminate();
}

auto kvf::create_window(glm::ivec2 const size, klib::CString const title, bool const decorated) -> UniqueWindow {
	glfw_init();
	glfwWindowHint(GLFW_DECORATED, decorated ? GLFW_TRUE : GLFW_FALSE);
	auto* window = glfwCreateWindow(size.x, size.y, title.c_str(), nullptr, nullptr);
	if (window == nullptr) { throw Error{"Failed to create GLFW Window"}; }
	glfwSetWindowSize(window, size.x, size.y);
	return UniqueWindow{window};
}

auto kvf::create_fullscreen_window(klib::CString const title, GLFWmonitor* target) -> UniqueWindow {
	glfw_init();
	if (target == nullptr) { target = glfwGetPrimaryMonitor(); }
	auto const* mode = glfwGetVideoMode(target);
	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
	glfwWindowHint(GLFW_CENTER_CURSOR, GLFW_TRUE);
	auto* window = glfwCreateWindow(mode->width, mode->height, title.c_str(), target, nullptr);
	if (window == nullptr) { throw Error{"Failed to create GLFW Window"}; }
	return UniqueWindow{window};
}

// render_device

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <kvf/device_block.hpp>
#include <kvf/render_device.hpp>
#include <kvf/util.hpp>

namespace kvf {
namespace {
constexpr auto srgb_formats_v = std::array{vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb, vk::Format::eA8B8G8R8SrgbPack32};
constexpr auto linear_formats_v = std::array{vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8A8Unorm, vk::Format::eA8B8G8R8UnormPack32};

constexpr auto is_srgb(vk::Format const format) -> bool { return std::ranges::find(srgb_formats_v, format) != srgb_formats_v.end(); }

[[nodiscard]] auto instance_extensions() -> std::span<char const* const> {
	auto count = std::uint32_t{};
	auto const* first = glfwGetRequiredInstanceExtensions(&count);
	return {first, count};
}

[[nodiscard]] auto filter_modes(std::span<vk::PresentModeKHR const> all) -> std::vector<vk::PresentModeKHR> {
	auto ret = std::vector<vk::PresentModeKHR>{};
	ret.reserve(all.size());
	for (auto const in : all) {
		if (std::ranges::find(RenderDevice::present_modes_v, in) == RenderDevice::present_modes_v.end()) { continue; }
		ret.push_back(in);
	}
	return ret;
}

[[nodiscard]] constexpr auto optimal_present_mode(std::span<vk::PresentModeKHR const> present_modes) {
	constexpr auto desired_v = std::array{vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eFifoRelaxed};
	for (auto const desired : desired_v) {
		if (std::ranges::find(present_modes, desired) != present_modes.end()) { return desired; }
	}
	return vk::PresentModeKHR::eFifo;
}

[[nodiscard]] constexpr auto compatible_surface_format(std::span<vk::SurfaceFormatKHR const> supported, bool const linear) -> vk::SurfaceFormatKHR {
	auto const& desired = linear ? linear_formats_v : srgb_formats_v;
	for (auto const srgb_format : desired) {
		auto const it = std::ranges::find_if(supported, [srgb_format](vk::SurfaceFormatKHR const& format) {
			return format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear && format.format == srgb_format;
		});
		if (it != supported.end()) { return *it; }
	}
	return vk::SurfaceFormatKHR{};
}

auto optimal_depth_format(vk::PhysicalDevice const& gpu) -> vk::Format {
	static constexpr auto target_v{vk::Format::eD32Sfloat};
	auto const props = gpu.getFormatProperties(target_v);
	if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) { return target_v; }
	return vk::Format::eD16Unorm;
}

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
		throw Error{"Invalid GPU"};
	}
};

class DearImGui {
  public:
	struct CreateInfo { // NOLINT(cppcoreguidelines-pro-type-member-init)
		GLFWwindow* window;
		vk::Instance instance;
		vk::Device device;
		vk::PhysicalDevice physical_device;
		std::uint32_t queue_family;
		vk::Queue queue;
		vk::SampleCountFlagBits samples;
		vk::PipelineRenderingCreateInfo prci;
		bool srgb_target;
	};

	DearImGui(DearImGui const&) = delete;
	DearImGui(DearImGui&&) = delete;
	auto operator=(DearImGui const&) = delete;
	auto operator=(DearImGui&&) = delete;

	DearImGui() = default;

	void init(CreateInfo const& create_info) {
		m_device = create_info.device;
		static constexpr std::uint32_t max_textures_v{16};
		auto const pool_sizes = std::array{
			vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, max_textures_v},
		};
		auto dpci = vk::DescriptorPoolCreateInfo{};
		dpci.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
		dpci.maxSets = max_textures_v;
		dpci.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
		dpci.pPoolSizes = pool_sizes.data();
		m_pool = m_device.createDescriptorPoolUnique(dpci);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		// ImGuiIO& io = ImGui::GetIO();
		// io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
		// io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

		ImGui::StyleColorsDark();
		if (create_info.srgb_target) {
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
			for (auto& colour : ImGui::GetStyle().Colors) {
				auto const linear = glm::convertSRGBToLinear(glm::vec4{colour.x, colour.y, colour.z, colour.w});
				colour = ImVec4{linear.x, linear.y, linear.z, linear.w};
			}
			ImGui::GetStyle().Colors[ImGuiCol_WindowBg].w = 0.99f; // more opaque
		}

		auto load_vk_func = +[](char const* name, void* user_data) {
			if (std::string_view{name} == "vkCmdBeginRenderingKHR") { name = "vkCmdBeginRendering"; }
			if (std::string_view{name} == "vkCmdEndRenderingKHR") { name = "vkCmdEndRendering"; }
			return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(*static_cast<vk::Instance*>(user_data), name);
		};
		auto instance = create_info.instance;
		ImGui_ImplVulkan_LoadFunctions(load_vk_func, &instance);
		ImGui_ImplGlfw_InitForVulkan(create_info.window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = create_info.instance;
		init_info.PhysicalDevice = create_info.physical_device;
		init_info.Device = create_info.device;
		init_info.QueueFamily = create_info.queue_family;
		init_info.Queue = create_info.queue;
		init_info.DescriptorPool = *m_pool;
		init_info.Subpass = 0;
		init_info.MinImageCount = 2;
		init_info.ImageCount = resource_buffering_v;
		init_info.MSAASamples = static_cast<VkSampleCountFlagBits>(create_info.samples);
		init_info.UseDynamicRendering = true;
		init_info.PipelineRenderingCreateInfo = create_info.prci;

		ImGui_ImplVulkan_Init(&init_info);
		ImGui_ImplVulkan_CreateFontsTexture();
	}

	~DearImGui() {
		if (!m_pool) { return; }
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void new_frame() { // NOLINT(misc-no-recursion)
		if (m_state == State::eEndFrame) { end_frame(); }
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		m_state = State::eEndFrame;
	}

	void end_frame() { // NOLINT(misc-no-recursion)
		// ImGui::Render calls ImGui::EndFrame
		if (m_state == State::eNewFrame) { new_frame(); }
		ImGui::Render();
		m_state = State::eNewFrame;
	}

	void render(vk::CommandBuffer const command_buffer) {
		if (m_state == State::eEndFrame) { end_frame(); }
		if (auto* data = ImGui::GetDrawData()) { ImGui_ImplVulkan_RenderDrawData(data, command_buffer); }
	}

	enum class State : std::int8_t { eNewFrame, eEndFrame };

	vk::Device m_device{};

	vk::UniqueDescriptorPool m_pool{};
	State m_state{};
};

struct MakeImageView {
	vk::Image image;
	vk::Format format;

	vk::ImageSubresourceRange subresource{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageViewType type{vk::ImageViewType::e2D};

	[[nodiscard]] auto operator()(vk::Device device) const -> vk::UniqueImageView {
		auto ivci = vk::ImageViewCreateInfo{};
		ivci.viewType = type;
		ivci.format = format;
		ivci.subresourceRange = subresource;
		ivci.image = image;
		return device.createImageViewUnique(ivci);
	}
};

struct Swapchain {
	void init(vk::Device device, vk::PhysicalDevice physical_device, vk::SwapchainCreateInfoKHR const& info, vk::Queue queue) {
		m_device = device;
		m_physical_device = physical_device;
		m_info = info;
		m_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
		m_info.imageArrayLayers = 1u;

		m_queue = queue;
	}

	void recreate(vk::Extent2D const framebuffer, std::optional<vk::PresentModeKHR> present_mode = {}) {
		if (framebuffer.width == 0 || framebuffer.height == 0) { return; }

		auto const surface_capabilities = m_physical_device.getSurfaceCapabilitiesKHR(m_info.surface);
		m_info.imageExtent = Swapchain::get_image_extent(surface_capabilities, framebuffer);
		if (present_mode) { m_info.presentMode = *present_mode; }
		m_info.minImageCount = Swapchain::get_image_count(surface_capabilities);
		m_info.oldSwapchain = *m_swapchain;

		m_device.waitIdle();
		m_swapchain = m_device.createSwapchainKHRUnique(m_info);
		if (!m_swapchain) { throw Error{"Failed to create Vulkan Swapchain"}; }

		auto image_count = std::uint32_t{};
		if (m_device.getSwapchainImagesKHR(*m_swapchain, &image_count, nullptr) != vk::Result::eSuccess) { throw Error{"Failed to get Swapchain Images"}; }
		m_images.resize(image_count);
		if (m_device.getSwapchainImagesKHR(*m_swapchain, &image_count, m_images.data()) != vk::Result::eSuccess) {
			throw Error{"Failed to get Swapchain Images"};
		}

		m_image_views.clear();
		m_image_views.reserve(m_images.size());
		auto make_image_view = MakeImageView{
			.image = vk::Image{},
			.format = m_info.imageFormat,
			.subresource = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
			.type = vk::ImageViewType::e2D,
		};
		for (auto const image : m_images) {
			make_image_view.image = image;
			m_image_views.push_back(make_image_view(m_device));
		}

		m_image_index.reset();
		m_layout = vk::ImageLayout::eUndefined;

		auto const extent = m_info.imageExtent;
		std::string_view const color_space = is_srgb(m_info.imageFormat) ? "sRGB" : "Linear";
		log::debug("Swapchain color-space: {}, extent: {}x{}, mode: {}", color_space, extent.width, extent.height, util::to_str(m_info.presentMode));
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
		default: throw Error{"Failed to acquire Swapchain Image"};
		}
	}

	auto present(vk::Queue queue, vk::Semaphore const wait) -> bool {
		KLIB_ASSERT(m_image_index);

		auto pi = vk::PresentInfoKHR{};
		pi.setImageIndices(*m_image_index).setSwapchains(*m_swapchain).setWaitSemaphores(wait);
		auto const result = queue.presentKHR(&pi);
		m_image_index.reset();

		switch (result) {
		case vk::Result::eErrorOutOfDateKHR: return false;
		case vk::Result::eSuboptimalKHR:
		case vk::Result::eSuccess: return true;
		default: throw Error{"Failed to present Swapchain Image"};
		}
	}

	[[nodiscard]] auto get_info() const -> vk::SwapchainCreateInfoKHR const& { return m_info; }
	[[nodiscard]] auto get_image() const -> vk::Image { return m_image_index ? m_images[*m_image_index] : vk::Image{}; }
	[[nodiscard]] auto get_image_view() const -> vk::ImageView { return m_image_index ? *m_image_views[*m_image_index] : vk::ImageView{}; }

  private:
	[[nodiscard]] static constexpr auto get_image_extent(vk::SurfaceCapabilitiesKHR const& caps, vk::Extent2D framebuffer) -> vk::Extent2D {
		constexpr auto limitless_v = std::numeric_limits<std::uint32_t>::max();
		if (caps.currentExtent.width < limitless_v && caps.currentExtent.height < limitless_v) { return caps.currentExtent; }
		auto const x = std::clamp(framebuffer.width, caps.minImageExtent.width, caps.maxImageExtent.width);
		auto const y = std::clamp(framebuffer.height, caps.minImageExtent.height, caps.maxImageExtent.height);
		return vk::Extent2D{x, y};
	}

	[[nodiscard]] static constexpr auto get_image_count(vk::SurfaceCapabilitiesKHR const& caps) -> std::uint32_t {
		if (caps.maxImageCount < caps.minImageCount) { return std::max(3u, caps.minImageCount + 1); }
		return std::clamp(3u, caps.minImageCount + 1, caps.maxImageCount);
	}

	vk::PhysicalDevice m_physical_device{};
	vk::Device m_device{};
	std::vector<vk::PresentModeKHR> m_present_modes{};

	vk::SwapchainCreateInfoKHR m_info{};
	vk::UniqueSwapchainKHR m_swapchain{};
	std::vector<vk::Image> m_images{};
	std::vector<vk::UniqueImageView> m_image_views{};

	std::optional<std::uint32_t> m_image_index{};
	vk::ImageLayout m_layout{};

	vk::Queue m_queue;
};

struct DescriptorAllocator {
	void reset() {
		for (auto& pool : m_pools) { device.resetDescriptorPool(*pool); }
		m_index = 0;
	}

	auto allocate(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> bool {
		KLIB_ASSERT(device && !pool_sizes.empty() && sets_per_pool > 0);
		if (layouts.empty() || out_sets.size() != layouts.size()) { return false; }
		auto result = try_allocate(out_sets, layouts);
		switch (result) {
		case vk::Result::eSuccess: return true;
		case vk::Result::eErrorOutOfPoolMemory: ++m_index; return try_allocate(out_sets, layouts) == vk::Result::eSuccess;
		default: return false;
		}
	}

	vk::Device device{};
	std::span<vk::DescriptorPoolSize const> pool_sizes{};
	std::uint32_t sets_per_pool{};

  private:
	auto get_pool() -> vk::DescriptorPool {
		while (m_index >= m_pools.size()) {
			auto dpci = vk::DescriptorPoolCreateInfo{};
			dpci.setPoolSizes(pool_sizes).setMaxSets(sets_per_pool);
			m_pools.push_back(device.createDescriptorPoolUnique(dpci));
		}
		return *m_pools.at(m_index);
	}

	auto try_allocate(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> vk::Result {
		auto const pool = get_pool();
		auto dsai = vk::DescriptorSetAllocateInfo{};
		dsai.setDescriptorPool(pool).setSetLayouts(layouts);
		return device.allocateDescriptorSets(&dsai, out_sets.data());
	}

	std::vector<vk::UniqueDescriptorPool> m_pools{};
	std::size_t m_index{};
};
} // namespace

struct RenderDevice::Impl {
	using Flag = RenderDeviceFlag;

	Impl(GLFWwindow* window, CreateInfo create_info)
		: m_window(window), m_flags(create_info.flags), m_pool_sizes(create_info.custom_pool_sizes.begin(), create_info.custom_pool_sizes.end()) {
		static auto const default_gpu_selector = GpuSelector{};
		if (create_info.gpu_selector == nullptr) { create_info.gpu_selector = &default_gpu_selector; }
		log::debug("kvf {}", klib::to_string(build_version_v));
		create_instance();
		create_surface();
		select_gpu(*create_info.gpu_selector);
		create_device();
		create_swapchain();
		create_allocator();
		init_set_allocators(create_info.sets_per_pool);

		m_imgui.new_frame();
	}

	[[nodiscard]] auto get_window() const -> GLFWwindow* { return m_window; }
	[[nodiscard]] auto get_flags() const -> Flag { return m_flags; }
	[[nodiscard]] auto get_frame_index() const -> FrameIndex { return FrameIndex{m_frame_index}; }

	[[nodiscard]] auto get_loader_api_version() const -> klib::Version { return m_loader_version; }
	[[nodiscard]] auto get_instance() const -> vk::Instance { return *m_instance; }
	[[nodiscard]] auto get_surface() const -> vk::SurfaceKHR { return *m_surface; }
	[[nodiscard]] auto get_gpu() const -> Gpu const& { return m_gpu; }
	[[nodiscard]] auto get_device() const -> vk::Device { return *m_device; }
	[[nodiscard]] auto get_queue_family() const -> std::uint32_t { return m_queue_family; }
	[[nodiscard]] auto get_allocator() const -> VmaAllocator { return m_allocator.get(); }

	[[nodiscard]] auto get_framebuffer_extent() const -> vk::Extent2D {
		auto width = int{};
		auto height = int{};
		glfwGetFramebufferSize(m_window, &width, &height);
		return {std::uint32_t(width), std::uint32_t(height)};
	}

	[[nodiscard]] auto get_present_mode() const -> vk::PresentModeKHR { return m_swapchain.get_info().presentMode; }
	[[nodiscard]] auto get_supported_present_modes() const -> std::span<vk::PresentModeKHR const> { return m_present_modes; }

	auto set_present_mode(vk::PresentModeKHR desired) -> bool {
		if (std::ranges::find(m_present_modes, desired) == m_present_modes.end()) { return false; }
		m_swapchain.recreate(get_framebuffer_extent(), desired);
		return true;
	}

	[[nodiscard]] auto get_swapchain_format() const -> vk::Format { return m_swapchain.get_info().imageFormat; }
	[[nodiscard]] auto get_depth_format() const -> vk::Format { return m_depth_format; }

	[[nodiscard]] auto image_barrier(vk::ImageAspectFlags const aspect = vk::ImageAspectFlagBits::eColor) const -> vk::ImageMemoryBarrier2 {
		auto ret = vk::ImageMemoryBarrier2{};
		ret.srcQueueFamilyIndex = ret.dstQueueFamilyIndex = get_queue_family();
		ret.subresourceRange.setAspectMask(aspect).setLevelCount(1).setLayerCount(1);
		return ret;
	}

	[[nodiscard]] auto sampler_info(vk::SamplerAddressMode wrap, vk::Filter filter, float aniso) const -> vk::SamplerCreateInfo {
		aniso = std::min(aniso, m_gpu.properties.limits.maxSamplerAnisotropy);
		auto ret = vk::SamplerCreateInfo{};
		ret.setAddressModeU(wrap)
			.setAddressModeV(wrap)
			.setAddressModeW(wrap)
			.setAnisotropyEnable(aniso > 0.0f ? vk::True : vk::False)
			.setMaxAnisotropy(aniso)
			.setMinFilter(filter)
			.setMagFilter(filter)
			.setMaxLod(VK_LOD_CLAMP_NONE)
			.setBorderColor(vk::BorderColor::eFloatTransparentBlack)
			.setMipmapMode(vk::SamplerMipmapMode::eNearest);
		return ret;
	}

	[[nodiscard]] auto create_pipeline(vk::PipelineLayout const layout, PipelineState const& state, PipelineFormat const format) const -> vk::UniquePipeline {
		auto shader_stages = std::array<vk::PipelineShaderStageCreateInfo, 2>{};
		shader_stages[0].setStage(vk::ShaderStageFlagBits::eVertex).setPName("main").setModule(state.vertex_shader);
		shader_stages[1].setStage(vk::ShaderStageFlagBits::eFragment).setPName("main").setModule(state.fragment_shader);

		auto pvisci = vk::PipelineVertexInputStateCreateInfo{};
		pvisci.setVertexAttributeDescriptions(state.vertex_attributes).setVertexBindingDescriptions(state.vertex_bindings);

		auto prsci = vk::PipelineRasterizationStateCreateInfo{};
		prsci.setPolygonMode(state.polygon_mode).setCullMode(state.cull_mode);

		auto pdssci = vk::PipelineDepthStencilStateCreateInfo{};
		auto const depth_test = (state.flags & PipelineFlag::DepthTest) == PipelineFlag::DepthTest;
		pdssci.setDepthTestEnable(depth_test ? vk::True : vk::False).setDepthCompareOp(state.depth_compare);

		auto const piasci = vk::PipelineInputAssemblyStateCreateInfo{{}, state.topology};

		auto pcbas = vk::PipelineColorBlendAttachmentState{};
		auto const alpha_blend = (state.flags & PipelineFlag::AlphaBlend) == PipelineFlag::AlphaBlend;
		using CCF = vk::ColorComponentFlagBits;
		pcbas.setColorWriteMask(CCF::eR | CCF::eG | CCF::eB | CCF::eA)
			.setBlendEnable(alpha_blend ? vk::True : vk::False)
			.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
			.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
			.setColorBlendOp(vk::BlendOp::eAdd)
			.setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
			.setDstAlphaBlendFactor(vk::BlendFactor::eZero)
			.setAlphaBlendOp(vk::BlendOp::eAdd);
		auto pcbsci = vk::PipelineColorBlendStateCreateInfo{};
		pcbsci.setAttachments(pcbas);

		auto const pdscis = std::array{
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor,
			vk::DynamicState::eLineWidth,
		};
		auto pdsci = vk::PipelineDynamicStateCreateInfo{};
		pdsci.setDynamicStates(pdscis);

		auto const pvsci = vk::PipelineViewportStateCreateInfo({}, 1, {}, 1);

		auto pmsci = vk::PipelineMultisampleStateCreateInfo{};
		pmsci.setRasterizationSamples(format.samples).setSampleShadingEnable(vk::False);

		auto prci = vk::PipelineRenderingCreateInfo{};
		if (format.color != vk::Format::eUndefined) { prci.setColorAttachmentFormats(format.color); }
		prci.setDepthAttachmentFormat(format.depth);

		auto gpci = vk::GraphicsPipelineCreateInfo{};
		gpci.setPVertexInputState(&pvisci)
			.setStages(shader_stages)
			.setPRasterizationState(&prsci)
			.setPDepthStencilState(&pdssci)
			.setPInputAssemblyState(&piasci)
			.setPColorBlendState(&pcbsci)
			.setPDynamicState(&pdsci)
			.setPViewportState(&pvsci)
			.setPMultisampleState(&pmsci)
			.setLayout(layout)
			.setPNext(&prci);

		auto const device = get_device();
		auto ret = vk::Pipeline{};
		if (device.createGraphicsPipelines({}, 1, &gpci, {}, &ret) != vk::Result::eSuccess) { return {}; }

		return vk::UniquePipeline{ret, device};
	}

	auto allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> bool {
		return m_set_allocators.at(m_frame_index).allocate(out_sets, layouts);
	}

	void queue_submit(vk::SubmitInfo2 const& si, vk::Fence const fence) {
		auto lock = std::scoped_lock{m_queue_mutex};
		m_queue.submit2(si, fence);
	}

	[[nodiscard]] auto is_window_closing() const -> bool { return glfwWindowShouldClose(m_window) == GLFW_TRUE; }

	void set_window_closing(bool const value) const { glfwSetWindowShouldClose(m_window, value ? GLFW_TRUE : GLFW_FALSE); }

	auto next_frame() -> vk::CommandBuffer {
		auto const drawn = *m_syncs.at(m_frame_index).drawn;
		if (!util::wait_for_fence(*m_device, drawn)) { throw Error{"Failed to wait for Render Fence"}; }

		glfwPollEvents();
		m_imgui.new_frame();
		m_set_allocators.at(m_frame_index).reset();

		m_current_cmd = &m_command_buffers.at(m_frame_index);
		m_current_cmd->begin();
		return m_current_cmd->cmd;
	}

	void render(RenderTarget const& frame, vk::Filter const filter) {
		m_imgui.end_frame();
		if (m_current_cmd == nullptr) { return; }

		auto const framebuffer_extent = get_framebuffer_extent();
		if (framebuffer_extent.width == 0 || framebuffer_extent.height == 0) { return; }

		if (m_swapchain.get_info().imageExtent != framebuffer_extent) { m_swapchain.recreate(framebuffer_extent); }

		auto const& sync = m_syncs.at(m_frame_index);
		if (!util::wait_for_fence(*m_device, *sync.drawn)) { throw Error{"Failed to wait for Render Fence"}; }

		auto lock = std::unique_lock{m_queue_mutex};
		if (!m_swapchain.acquire_next_image(*sync.draw)) { // out of date
			lock.unlock();
			m_swapchain.recreate(framebuffer_extent);
			m_current_cmd->end();
			m_current_cmd = {};
			return;
		}
		lock.unlock();

		m_device->resetFences(*sync.drawn); // must submit after reset

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
			util::record_barrier(m_current_cmd->cmd, barrier);
			blit_to_backbuffer(frame, backbuffer, m_current_cmd->cmd, filter);
			backbuffer_load_op = vk::AttachmentLoadOp::eLoad;
		}

		if (should_render_imgui) {
			barrier = transition_backbuffer(backbuffer.image, vk::ImageLayout::eAttachmentOptimal);
			util::record_barrier(m_current_cmd->cmd, barrier);
			auto cai = vk::RenderingAttachmentInfo{};
			cai.setImageView(backbuffer.view)
				.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
				.setLoadOp(backbuffer_load_op)
				.setStoreOp(vk::AttachmentStoreOp::eStore);
			render_imgui(m_current_cmd->cmd, cai, backbuffer.extent);
		}

		barrier = transition_backbuffer(backbuffer.image, vk::ImageLayout::ePresentSrcKHR);
		util::record_barrier(m_current_cmd->cmd, barrier);

		m_current_cmd->end();

		auto const cbsi = vk::CommandBufferSubmitInfo{m_current_cmd->cmd};
		auto const wssi = vk::SemaphoreSubmitInfo{*sync.draw, 0, vk::PipelineStageFlagBits2::eTopOfPipe};
		auto const sssi = vk::SemaphoreSubmitInfo{*sync.present, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput};
		auto si = vk::SubmitInfo2{};
		si.setCommandBufferInfos(cbsi).setWaitSemaphoreInfos(wssi).setSignalSemaphoreInfos(sssi);

		lock.lock();
		m_queue.submit2(si, *sync.drawn);
		auto const present_sucess = m_swapchain.present(m_queue, *sync.present);
		lock.unlock();

		if (!present_sucess) { m_swapchain.recreate(get_framebuffer_extent()); }

		m_frame_index = (m_frame_index + 1) % resource_buffering_v;
		m_current_cmd = nullptr;
	}

	bool should_render_imgui{true};

  private:
	struct Sync {
		vk::UniqueSemaphore draw{};
		vk::UniqueSemaphore present{};
		vk::UniqueFence drawn{};
	};

	struct RenderCmd {
		vk::CommandBuffer cmd{};
		bool recording{};

		void begin() {
			if (recording) { end(); }
			cmd.begin(vk::CommandBufferBeginInfo{});
			recording = true;
		}

		void end() {
			if (!recording) { return; }
			cmd.end();
			recording = false;
		}
	};

	struct Deleter {
		void operator()(VmaAllocator allocator) const noexcept { vmaDestroyAllocator(allocator); }
	};

	void create_instance() {
		VULKAN_HPP_DEFAULT_DISPATCHER.init();

		static auto const min_ver_str = klib::to_string(vk_api_version_v);
		auto const vk_api_version = vk::enumerateInstanceVersion();
		m_loader_version = klib::Version{
			.major = VK_VERSION_MAJOR(vk_api_version),
			.minor = VK_VERSION_MINOR(vk_api_version),
			.patch = VK_VERSION_PATCH(vk_api_version),
		};
		log::debug("Vulkan loader (Instance API) version: {}", klib::to_string(m_loader_version));

		auto app_info = vk::ApplicationInfo{};
		app_info.apiVersion = VK_MAKE_VERSION(vk_api_version_v.major, vk_api_version_v.minor, vk_api_version_v.patch);
		auto ici = vk::InstanceCreateInfo{};
		ici.pApplicationInfo = &app_info;
		auto const wsi_extensions = instance_extensions();
		auto extensions = std::vector(wsi_extensions.begin(), wsi_extensions.end());

		auto const validation_enables = std::array{vk::ValidationFeatureEnableEXT::eSynchronizationValidation};
		auto validation_features = vk::ValidationFeaturesEXT{};
		validation_features.setEnabledValidationFeatures(validation_enables);

		if ((m_flags & Flag::ValidationLayers) == Flag::ValidationLayers) {
			static constexpr char const* validation_layer_v = "VK_LAYER_KHRONOS_validation";
			auto const props = vk::enumerateInstanceLayerProperties();
			static constexpr auto pred = [](vk::LayerProperties const& p) { return p.layerName == std::string_view{validation_layer_v}; };
			auto const it = std::ranges::find_if(props, pred);
			if (it == props.end()) {
				log::warn("Validation layers requested but {} is not available", validation_layer_v);
				m_flags &= ~Flag::ValidationLayers;
			} else {
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
				ici.setPEnabledLayerNames(validation_layer_v);
				ici.pNext = &validation_features;
			}
		}

		ici.enabledExtensionCount = std::uint32_t(extensions.size());
		ici.ppEnabledExtensionNames = extensions.data();
		try {
			m_instance = vk::createInstanceUnique(ici);
		} catch (vk::LayerNotPresentError const& e) {
			log::error("{}", e.what());
			ici.enabledLayerCount = 0;
			m_instance = vk::createInstanceUnique(ici);
		}

		if (!m_instance) { throw Error{"Failed to create Vulkan Instance"}; }
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_instance);
		log::debug("Vulkan {} Instance created", min_ver_str);

		if ((m_flags & Flag::ValidationLayers) == Flag::ValidationLayers) { create_debug_messenger(); }
	}

	void create_debug_messenger() {
		auto dmci = vk::DebugUtilsMessengerCreateInfoEXT{};
		using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
		using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
		using Types = vk::DebugUtilsMessageTypeFlagsEXT;
		using Data = vk::DebugUtilsMessengerCallbackDataEXT;
		auto severity_v = Severity::eError | Severity::eWarning;
		if constexpr (klib::debug_v) { severity_v |= Severity::eInfo; }
		static constexpr auto types_v = Type::eGeneral | Type::ePerformance | Type::eValidation | Type::eDeviceAddressBinding;
		static auto const on_msg = [](Severity severity, Types /*unused*/, Data const* data, void* /*unused*/) -> vk::Bool32 {
			static constexpr std::string_view tag_v{"validation"};
			switch (severity) {
			case Severity::eError: klib::log::error(tag_v, "{}", data->pMessage); break;
			case Severity::eWarning: klib::log::warn(tag_v, "{}", data->pMessage); break;
			case Severity::eInfo: klib::log::info(tag_v, "{}", data->pMessage); break;
			default: break;
			}
			return vk::False;
		};
		dmci.setMessageSeverity(severity_v).setMessageType(types_v).setPfnUserCallback(on_msg);
		m_debug_messenger = m_instance->createDebugUtilsMessengerEXTUnique(dmci);
		log::debug("Vulkan Debug Messenger created");
	}

	void create_surface() {
		VkSurfaceKHR surface{};
		glfwCreateWindowSurface(*m_instance, m_window, nullptr, &surface);
		if (surface == VK_NULL_HANDLE) { throw Error{"Failed to create Vulkan Surface"}; }
		m_surface = vk::UniqueSurfaceKHR{surface, *m_instance};
	}

	void select_gpu(GpuSelector const& selector) {
		auto const devices = m_instance->enumeratePhysicalDevices();
		auto list = GpuList::get_viable(*m_instance, *m_surface);
		if (list.gpus.empty()) { throw Error{"No viable GPUs"}; }
		auto const* selected = &selector.select(list.gpus);
		m_queue_family = list.get_queue_family(selected);
		m_gpu = *selected;
		m_depth_format = optimal_depth_format(m_gpu.device);
		log::debug("Using GPU: {}, queue family: {}", m_gpu.properties.deviceName.data(), m_queue_family);
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

		auto sync_feature = vk::PhysicalDeviceSynchronization2Features{vk::True};
		auto dr_feature = vk::PhysicalDeviceDynamicRenderingFeatures{vk::True};
		sync_feature.pNext = &dr_feature;

		auto dci = vk::DeviceCreateInfo{};
		static constexpr auto extensions_v = std::array{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
		dci.setEnabledExtensionCount(std::uint32_t(extensions_v.size()))
			.setPEnabledExtensionNames(extensions_v)
			.setQueueCreateInfos(qci)
			.setPEnabledFeatures(&enabled_features)
			.setPNext(&sync_feature);

		m_device = m_gpu.device.createDeviceUnique(dci);
		if (!m_device) { throw Error{"Failed to create Vulkan Device"}; }
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_device);

		m_queue = m_device->getQueue(m_queue_family, 0);
		log::debug("Vulkan Device created");

		m_device_block.get() = *m_device;
	}

	void create_swapchain() {
		auto const linear_backbuffer = (m_flags & Flag::LinearBackbuffer) == Flag::LinearBackbuffer;
		auto const surface_format = compatible_surface_format(m_gpu.device.getSurfaceFormatsKHR(*m_surface), linear_backbuffer);
		m_present_modes = filter_modes(m_gpu.device.getSurfacePresentModesKHR(*m_surface));
		auto sci = vk::SwapchainCreateInfoKHR{};
		sci.surface = *m_surface;
		sci.presentMode = optimal_present_mode(m_present_modes);
		sci.imageFormat = surface_format.format;
		sci.queueFamilyIndexCount = 1u;
		sci.pQueueFamilyIndices = &m_queue_family;
		sci.imageColorSpace = surface_format.colorSpace;
		m_swapchain.init(*m_device, m_gpu.device, sci, m_queue);
		m_swapchain.recreate(get_framebuffer_extent());

		auto const cpci = vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_queue_family};
		m_command_pool = m_device->createCommandPoolUnique(cpci);
		auto const cbai = vk::CommandBufferAllocateInfo{*m_command_pool, vk::CommandBufferLevel::ePrimary, std::uint32_t(resource_buffering_v)};
		auto cmds = Buffered<vk::CommandBuffer>{};
		if (m_device->allocateCommandBuffers(&cbai, cmds.data()) != vk::Result::eSuccess) { throw Error{"Failed to allocate render CommandBuffer(s)"}; }
		for (auto [cmd, render_cmd] : std::views::zip(cmds, m_command_buffers)) { render_cmd.cmd = cmd; }

		for (auto& sync : m_syncs) {
			sync.draw = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
			sync.present = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
			sync.drawn = m_device->createFenceUnique(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
		}

		auto prci = vk::PipelineRenderingCreateInfo{};
		prci.setColorAttachmentCount(1).setColorAttachmentFormats(m_swapchain.get_info().imageFormat);
		auto const dici = DearImGui::CreateInfo{
			.window = m_window,
			.instance = *m_instance,
			.device = *m_device,
			.physical_device = m_gpu.device,
			.queue_family = m_queue_family,
			.queue = m_queue,
			.samples = vk::SampleCountFlagBits::e1,
			.prci = prci,
			.srgb_target = !linear_backbuffer,
		};
		m_imgui.init(dici);
		log::debug("Dear ImGui initialized");
	}

	void create_allocator() {
		auto vaci = VmaAllocatorCreateInfo{};
		vaci.instance = *m_instance;
		vaci.physicalDevice = m_gpu.device;
		vaci.device = *m_device;
		auto dl = VULKAN_HPP_DEFAULT_DISPATCHER;
		auto vkFunc = VmaVulkanFunctions{};
		vkFunc.vkGetInstanceProcAddr = dl.vkGetInstanceProcAddr;
		vkFunc.vkGetDeviceProcAddr = dl.vkGetDeviceProcAddr;
		vaci.pVulkanFunctions = &vkFunc;
		if (vmaCreateAllocator(&vaci, &m_allocator.get()) != VK_SUCCESS) { throw Error{"Failed to create Vulkan Allocator"}; }
		log::debug("Vulkan Allocator created");
	}

	void init_set_allocators(std::uint32_t const sets_per_pool) {
		if (m_pool_sizes.empty()) {
			static constexpr auto descriptors_per_type_v{8};
			m_pool_sizes = {
				vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, descriptors_per_type_v},
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, descriptors_per_type_v},
				vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, descriptors_per_type_v},
			};
		}
		for (auto& allocator : m_set_allocators) {
			allocator.device = *m_device;
			allocator.pool_sizes = m_pool_sizes;
			allocator.sets_per_pool = sets_per_pool;
		}
	}

	void blit_to_backbuffer(RenderTarget const& frame, RenderTarget const& backbuffer, vk::CommandBuffer cmd, vk::Filter filter) const {
		static auto const isr_v = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		auto barrier = image_barrier();
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

	void render_imgui(vk::CommandBuffer cmd, vk::RenderingAttachmentInfo const& backbuffer, vk::Extent2D const extent) {
		auto render_area = vk::Rect2D{};
		render_area.setExtent(extent);

		auto ri = vk::RenderingInfo{};
		ri.setColorAttachments(backbuffer).setLayerCount(1).setRenderArea(render_area);
		cmd.beginRendering(ri);
		m_imgui.render(cmd);
		cmd.endRendering();
	}

	[[nodiscard]] auto transition_backbuffer(vk::Image backbuffer, vk::ImageLayout const target) -> vk::ImageMemoryBarrier2 {
		static constexpr auto isr_v = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		auto ret = image_barrier();
		ret.setImage(backbuffer).setSubresourceRange(isr_v).setOldLayout(m_backbuffer_layout).setNewLayout(target);
		switch (m_backbuffer_layout) {
		case vk::ImageLayout::eUndefined:
		case vk::ImageLayout::ePresentSrcKHR: ret.setSrcAccessMask(vk::AccessFlagBits2::eNone).setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe); break;
		case vk::ImageLayout::eTransferDstOptimal:
			ret.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite).setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer);
			break;
		case vk::ImageLayout::eAttachmentOptimal:
			ret.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite)
				.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
			break;
		default: KLIB_ASSERT(false);
		}
		switch (target) {
		case vk::ImageLayout::eTransferDstOptimal:
			ret.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite).setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
			break;
		case vk::ImageLayout::eAttachmentOptimal:
			ret.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
			break;
		case vk::ImageLayout::ePresentSrcKHR:
			ret.setDstAccessMask(vk::AccessFlagBits2::eNone).setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
			break;
		default: KLIB_ASSERT(false);
		}

		m_backbuffer_layout = target;
		return ret;
	}

	GLFWwindow* m_window{};
	Flag m_flags{};

	klib::Version m_loader_version{};

	vk::UniqueInstance m_instance{};
	vk::UniqueDebugUtilsMessengerEXT m_debug_messenger{};
	vk::UniqueSurfaceKHR m_surface{};
	Gpu m_gpu{};
	std::uint32_t m_queue_family{};
	vk::Format m_depth_format{};

	vk::UniqueDevice m_device{};
	vk::Queue m_queue{};
	std::mutex m_queue_mutex{};

	std::vector<vk::PresentModeKHR> m_present_modes{};
	Swapchain m_swapchain{};
	Buffered<Sync> m_syncs{};
	vk::UniqueCommandPool m_command_pool{};
	Buffered<RenderCmd> m_command_buffers{};
	DearImGui m_imgui{};

	klib::Unique<VmaAllocator, Deleter> m_allocator{};
	std::vector<vk::DescriptorPoolSize> m_pool_sizes{};
	Buffered<DescriptorAllocator> m_set_allocators{};

	vk::ImageLayout m_backbuffer_layout{};
	std::size_t m_frame_index{};
	RenderCmd* m_current_cmd{};

	DeviceBlock m_device_block{};
};

void RenderDevice::Deleter::operator()(Impl* ptr) const noexcept { std::default_delete<Impl>{}(ptr); }

RenderDevice::RenderDevice(gsl::not_null<GLFWwindow*> window, CreateInfo create_info) : m_impl(new Impl(window, std::move(create_info))) {}

auto RenderDevice::get_window() const -> GLFWwindow* { return m_impl->get_window(); }
auto RenderDevice::get_flags() const -> RenderDeviceFlag { return m_impl->get_flags(); }
auto RenderDevice::get_frame_index() const -> FrameIndex { return m_impl->get_frame_index(); }

auto RenderDevice::get_loader_api_version() const -> klib::Version { return m_impl->get_loader_api_version(); }
auto RenderDevice::get_instance() const -> vk::Instance { return m_impl->get_instance(); }
auto RenderDevice::get_surface() const -> vk::SurfaceKHR { return m_impl->get_surface(); }
auto RenderDevice::get_gpu() const -> Gpu const& { return m_impl->get_gpu(); }
auto RenderDevice::get_device() const -> vk::Device { return m_impl->get_device(); }
auto RenderDevice::get_queue_family() const -> std::uint32_t { return m_impl->get_queue_family(); }
auto RenderDevice::get_allocator() const -> VmaAllocator { return m_impl->get_allocator(); }

auto RenderDevice::get_framebuffer_extent() const -> vk::Extent2D { return m_impl->get_framebuffer_extent(); }
auto RenderDevice::get_present_mode() const -> vk::PresentModeKHR { return m_impl->get_present_mode(); }
auto RenderDevice::get_supported_present_modes() const -> std::span<vk::PresentModeKHR const> { return m_impl->get_supported_present_modes(); }
auto RenderDevice::set_present_mode(vk::PresentModeKHR const desired) -> bool { return m_impl->set_present_mode(desired); }

auto RenderDevice::get_swapchain_format() const -> vk::Format { return m_impl->get_swapchain_format(); }
auto RenderDevice::get_depth_format() const -> vk::Format { return m_impl->get_depth_format(); }
auto RenderDevice::image_barrier(vk::ImageAspectFlags const aspect) const -> vk::ImageMemoryBarrier2 { return m_impl->image_barrier(aspect); }
auto RenderDevice::sampler_info(vk::SamplerAddressMode wrap, vk::Filter filter, float aniso) const -> vk::SamplerCreateInfo {
	return m_impl->sampler_info(wrap, filter, aniso);
}

auto RenderDevice::create_pipeline(vk::PipelineLayout layout, PipelineState const& state, PipelineFormat const format) const -> vk::UniquePipeline {
	return m_impl->create_pipeline(layout, state, format);
}

auto RenderDevice::allocate_sets(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> bool {
	return m_impl->allocate_sets(out_sets, layouts);
}

void RenderDevice::queue_submit(vk::SubmitInfo2 const& si, vk::Fence const fence) { m_impl->queue_submit(si, fence); }

auto RenderDevice::get_render_imgui() const -> bool { return m_impl->should_render_imgui; }
void RenderDevice::set_render_imgui(bool should_render) { m_impl->should_render_imgui = should_render; }

auto RenderDevice::is_window_closing() const -> bool { return m_impl->is_window_closing(); }
void RenderDevice::set_window_closing(bool const value) const { m_impl->set_window_closing(value); }

auto RenderDevice::next_frame() -> vk::CommandBuffer { return m_impl->next_frame(); }
void RenderDevice::render(RenderTarget const& frame, vk::Filter const filter) { m_impl->render(frame, filter); }
} // namespace kvf

// vma

#include <kvf/vma.hpp>

namespace kvf::vma {
void Buffer::Deleter::operator()(Payload const& buffer) const noexcept { vmaDestroyBuffer(buffer.allocator, buffer.resource, buffer.allocation); }

Buffer::Buffer(gsl::not_null<RenderDevice*> render_device, CreateInfo const& create_info, vk::DeviceSize size)
	: Resource<vk::Buffer>(render_device), m_info(create_info) {
	if (m_info.type == BufferType::Device) { m_info.usage |= vk::BufferUsageFlagBits::eTransferDst; }
	if (!resize(size)) { throw Error{"Failed to create Vulkan Buffer"}; }
}

auto Buffer::resize(vk::DeviceSize size) -> bool {
	if (m_device == nullptr) { return false; }
	ensure_positive(size);
	if (m_capacity >= size) {
		m_size = size;
		return true;
	}

	auto vaci = VmaAllocationCreateInfo{};
	vaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	if (m_info.type == BufferType::Device) {
		vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	} else {
		vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		vaci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
	}

	auto const bci = vk::BufferCreateInfo{{}, size, m_info.usage};
	auto vbci = static_cast<VkBufferCreateInfo>(bci);

	VmaAllocation allocation{};
	VkBuffer buffer{};
	auto alloc_info = VmaAllocationInfo{};
	if (vmaCreateBuffer(m_device->get_allocator(), &vbci, &vaci, &buffer, &allocation, &alloc_info) != VK_SUCCESS) { return false; }

	m_size = m_capacity = size;
	m_buffer = Payload{
		.allocator = m_device->get_allocator(),
		.allocation = allocation,
		.resource = buffer,
	};
	m_mapped = alloc_info.pMappedData;
	return true;
}

void Image::Deleter::operator()(Payload const& image) const noexcept { vmaDestroyImage(image.allocator, image.resource, image.allocation); }

Image::Image(gsl::not_null<RenderDevice*> render_device, CreateInfo const& create_info, vk::Extent2D extent)
	: Resource<vk::Image>(render_device), m_info(create_info) {
	m_info.usage |= CreateInfo::implicit_usage_v;
	if (!resize(extent)) { throw Error{"Failed to create Vulkan Image"}; }
}

auto Image::resize(vk::Extent2D extent) -> bool {
	if (m_device == nullptr) { return false; }
	ensure_positive(extent.width, extent.height);
	if (m_extent == extent) { return true; }

	auto const mip_mapped = (m_info.flags & ImageFlag::MipMapped) == ImageFlag::MipMapped;
	auto const queue_family = m_device->get_queue_family();
	auto ici = vk::ImageCreateInfo{};
	ici.setExtent({extent.width, extent.height, 1})
		.setFormat(m_info.format)
		.setUsage(m_info.usage)
		.setImageType(vk::ImageType::e2D)
		.setArrayLayers(m_info.layers)
		.setMipLevels(mip_mapped ? util::compute_mip_levels(extent) : 1)
		.setSamples(m_info.samples)
		.setTiling(vk::ImageTiling::eOptimal)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setQueueFamilyIndices(queue_family);
	auto const vici = static_cast<VkImageCreateInfo>(ici);
	auto vaci = VmaAllocationCreateInfo{};
	vaci.usage = VMA_MEMORY_USAGE_AUTO;
	if ((m_info.flags & ImageFlag::DedicatedAlloc) == ImageFlag::DedicatedAlloc) {
		vaci.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		vaci.priority = 1.0f;
	}
	VkImage image{};
	VmaAllocation allocation{};
	if (vmaCreateImage(m_device->get_allocator(), &vici, &vaci, &image, &allocation, {}) != VK_SUCCESS) { return false; }

	m_extent = extent;
	m_mip_levels = ici.mipLevels;
	m_image = Payload{
		.allocator = m_device->get_allocator(),
		.allocation = allocation,
		.resource = image,
	};
	auto const make_image_view = MakeImageView{
		.image = m_image.get().resource,
		.format = ici.format,
		.subresource = subresource_range(),
		.type = vk::ImageViewType::e2D,
	};
	m_view = make_image_view(m_device->get_device());
	m_layout = vk::ImageLayout::eUndefined;

	return true;
}

void Image::transition(vk::CommandBuffer command_buffer, vk::ImageMemoryBarrier2 barrier) {
	barrier.setImage(get_image())
		.setSrcQueueFamilyIndex(m_device->get_queue_family())
		.setDstQueueFamilyIndex(barrier.srcQueueFamilyIndex)
		.setSubresourceRange(subresource_range());
	util::record_barrier(command_buffer, barrier);
	m_layout = barrier.newLayout;
}

auto Image::subresource_range() const -> vk::ImageSubresourceRange { return vk::ImageSubresourceRange{m_info.aspect, 0, m_mip_levels, 0, m_info.layers}; }
} // namespace kvf::vma

// render_pass

#include <kvf/render_pass.hpp>

namespace kvf {
namespace {
constexpr auto is_norm(float const f) { return f >= 0.0f && f <= 1.0f; }
constexpr auto is_norm(glm::vec2 const v) { return is_norm(v.x) && is_norm(v.y); }
constexpr auto is_norm(UvRect const& r) { return is_norm(r.lt) && is_norm(r.rb); }
} // namespace

RenderPass::RenderPass(gsl::not_null<RenderDevice*> render_device, vk::SampleCountFlagBits const samples) : m_device(render_device), m_samples(samples) {}

auto RenderPass::set_color_target(vk::Format format) -> RenderPass& {
	if (format == vk::Format::eUndefined) { format = is_srgb(m_device->get_swapchain_format()) ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm; }
	auto const color_ici = vma::ImageCreateInfo{
		.format = format,
		.aspect = vk::ImageAspectFlagBits::eColor,
		.usage = vk::ImageUsageFlagBits::eColorAttachment,
		.samples = m_samples,
		.flags = vma::ImageFlag::DedicatedAlloc,
	};
	auto const resolve_ici = [&] {
		auto ret = color_ici;
		ret.samples = vk::SampleCountFlagBits::e1;
		return ret;
	}();
	for (auto& framebuffer : m_framebuffers) {
		framebuffer.color = vma::Image{m_device, color_ici, m_extent};
		if (m_samples > vk::SampleCountFlagBits::e1) { framebuffer.resolve = vma::Image{m_device, resolve_ici, m_extent}; }
	}
	return *this;
}

auto RenderPass::set_depth_target() -> RenderPass& {
	auto const depth_ici = vma::ImageCreateInfo{
		.format = m_device->get_depth_format(),
		.aspect = vk::ImageAspectFlagBits::eDepth,
		.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
		.samples = m_samples,
		.flags = vma::ImageFlag::DedicatedAlloc,
	};
	for (auto& framebuffer : m_framebuffers) { framebuffer.depth = vma::Image{m_device, depth_ici, m_extent}; }
	return *this;
}

auto RenderPass::create_pipeline(vk::PipelineLayout layout, PipelineState const& state) -> vk::UniquePipeline {
	auto const format = PipelineFormat{
		.samples = m_samples,
		.color = get_color_format(),
		.depth = get_depth_format(),
	};
	return m_device->create_pipeline(layout, state, format);
}

auto RenderPass::get_color_format() const -> vk::Format {
	if (!has_color_target()) { return vk::Format::eUndefined; }
	return m_framebuffers[0].color.get_info().format;
}

auto RenderPass::get_depth_format() const -> vk::Format {
	if (!has_depth_target()) { return vk::Format::eUndefined; }
	return m_framebuffers[0].depth.get_info().format;
}

auto RenderPass::render_target() const -> RenderTarget const& {
	if (m_targets.resolve.view) { return m_targets.resolve; }
	if (m_targets.color.view) { return m_targets.color; }
	return m_targets.depth;
}

void RenderPass::begin_render(vk::CommandBuffer const command_buffer, vk::Extent2D extent) {
	if (!has_color_target() && !has_depth_target()) { return; }

	ensure_positive(extent.width, extent.height);
	m_extent = extent;
	m_command_buffer = command_buffer;

	set_render_targets();

	m_barriers.clear();
	if (m_targets.color.image) {
		auto barrier = m_device->image_barrier();
		barrier.setImage(m_targets.color.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
		m_barriers.push_back(barrier);
	}
	if (m_targets.resolve.image) {
		KLIB_ASSERT(m_targets.color.image && !m_barriers.empty());
		auto barrier = m_barriers.back();
		barrier.setImage(m_targets.resolve.image);
		m_barriers.push_back(barrier);
	}
	if (m_targets.depth.image) {
		auto barrier = m_device->image_barrier(vk::ImageAspectFlagBits::eDepth);
		barrier.setImage(m_targets.depth.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
		m_barriers.push_back(barrier);
	}

	if (m_command_buffer) { util::record_barriers(m_command_buffer, m_barriers); }

	auto cai = vk::RenderingAttachmentInfo{};
	auto dai = vk::RenderingAttachmentInfo{};

	if (m_targets.color.view) {
		auto const cc = clear_color;
		cai.setImageView(m_targets.color.view)
			.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearColorValue{cc.x, cc.y, cc.z, cc.w});
	}
	if (m_targets.resolve.view) {
		cai.setResolveImageView(m_targets.resolve.view)
			.setResolveImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setResolveMode(vk::ResolveModeFlagBits::eAverage);
	}
	if (m_targets.depth.view) {
		dai.setImageView(m_targets.depth.view)
			.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(depth_store_op)
			.setClearValue(clear_depth);
	}

	auto ri = vk::RenderingInfo{};
	if (dai.imageView) { ri.setPDepthAttachment(&dai).setRenderArea(vk::Rect2D{{}, m_targets.depth.extent}); }
	if (cai.imageView) { ri.setColorAttachments(cai).setLayerCount(1).setRenderArea(vk::Rect2D{{}, m_targets.color.extent}); }
	if (m_command_buffer) { command_buffer.beginRendering(ri); }
}

void RenderPass::end_render() {
	if (m_command_buffer) { m_command_buffer.endRendering(); }

	m_barriers.clear();
	if (m_targets.color.image) {
		auto barrier = m_device->image_barrier();
		barrier.setImage(m_targets.color.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eTransferRead)
			.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eTransfer)
			.setOldLayout(vk::ImageLayout::eAttachmentOptimal)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		m_barriers.push_back(barrier);
	}
	if (m_targets.resolve.image) {
		KLIB_ASSERT(m_targets.color.image && !m_barriers.empty());
		auto barrier = m_barriers.back();
		barrier.setImage(m_targets.resolve.image);
		m_barriers.push_back(barrier);
	}
	if (m_targets.depth.image && depth_store_op == vk::AttachmentStoreOp::eStore) {
		auto barrier = m_device->image_barrier(vk::ImageAspectFlagBits::eDepth);
		barrier.setImage(m_targets.depth.image)
			.setSrcAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setOldLayout(vk::ImageLayout::eAttachmentOptimal)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		m_barriers.push_back(barrier);
	}
	if (m_command_buffer) { util::record_barriers(m_command_buffer, m_barriers); }

	m_command_buffer = vk::CommandBuffer{};
}

auto RenderPass::to_viewport(UvRect n_rect) const -> vk::Viewport {
	if (!is_norm(n_rect)) { n_rect = uv_rect_v; }
	auto const fb_size = util::to_glm_vec(get_extent());
	auto const rect = UvRect{.lt = n_rect.lt * fb_size, .rb = n_rect.rb * fb_size};
	auto const vp_size = rect.size();
	return vk::Viewport{rect.lt.x, rect.rb.y, vp_size.x, -vp_size.y};
}

auto RenderPass::to_scissor(UvRect n_rect) const -> vk::Rect2D {
	if (!is_norm(n_rect)) { n_rect = uv_rect_v; }
	auto const fb_size = kvf::util::to_glm_vec(get_extent());
	auto const rect = kvf::UvRect{.lt = n_rect.lt * fb_size, .rb = n_rect.rb * fb_size};
	auto const offset = glm::ivec2{rect.lt};
	auto const extent = glm::uvec2{rect.size()};
	return vk::Rect2D{vk::Offset2D{offset.x, offset.y}, vk::Extent2D{extent.x, extent.y}};
}

void RenderPass::bind_pipeline(vk::Pipeline const pipeline) const {
	if (!m_command_buffer) { return; }
	m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	m_command_buffer.setViewport(0, to_viewport(uv_rect_v));
	m_command_buffer.setScissor(0, to_scissor(uv_rect_v));
}

void RenderPass::set_render_targets() {
	auto& framebuffer = m_framebuffers.at(std::size_t(m_device->get_frame_index()));
	if (framebuffer.color && framebuffer.color.get_extent() != m_extent) {
		framebuffer.color.resize(m_extent);
		if (framebuffer.resolve) { framebuffer.resolve.resize(m_extent); }
	}
	if (framebuffer.depth && framebuffer.depth.get_extent() != m_extent) { framebuffer.depth.resize(m_extent); }

	m_targets.color = framebuffer.color.render_target();
	m_targets.resolve = framebuffer.resolve.render_target();
	m_targets.depth = framebuffer.depth.render_target();
}
} // namespace kvf

// command_buffer

#include <kvf/command_buffer.hpp>

namespace kvf {
CommandBuffer::CommandBuffer(gsl::not_null<RenderDevice*> render_device) : m_device(render_device) {
	auto const device = render_device->get_device();
	auto cpci = vk::CommandPoolCreateInfo{};
	cpci.setQueueFamilyIndex(render_device->get_queue_family()).setFlags(vk::CommandPoolCreateFlagBits::eTransient);
	m_pool = device.createCommandPoolUnique(cpci);
	auto cbai = vk::CommandBufferAllocateInfo{};
	cbai.setCommandPool(*m_pool).setCommandBufferCount(1);
	if (device.allocateCommandBuffers(&cbai, &m_cmd) != vk::Result::eSuccess) { throw Error{"Failed to allocate Vulkan Command Buffer"}; }
	m_cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
}

auto CommandBuffer::submit_and_wait(std::chrono::seconds const timeout) -> bool {
	m_cmd.end();
	auto const cbsi = vk::CommandBufferSubmitInfo{m_cmd};
	auto si = vk::SubmitInfo2{};
	si.setCommandBufferInfos(cbsi);
	auto const fence = m_device->get_device().createFenceUnique({});
	m_device->queue_submit(si, *fence);
	return util::wait_for_fence(m_device->get_device(), *fence, timeout);
}
} // namespace kvf

// image_bitmap

#include <stb/stb_image.h>
#include <kvf/image_bitmap.hpp>

namespace kvf {
void ImageBitmap::Deleter::operator()(Bitmap const& bitmap) const noexcept {
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
	stbi_image_free(const_cast<std::byte*>(bitmap.bytes.data()));
}

ImageBitmap::ImageBitmap(std::span<std::byte const> compressed) { decompress(compressed); }

auto ImageBitmap::decompress(std::span<std::byte const> compressed) -> bool {
	auto const* ptr = static_cast<void const*>(compressed.data());
	auto size = glm::ivec2{};
	auto in_channels = int{};
	void* result = stbi_load_from_memory(static_cast<stbi_uc const*>(ptr), int(compressed.size()), &size.x, &size.y, &in_channels, int(channels_v));
	if (result == nullptr || !is_positive(size)) { return false; }

	m_bitmap = Bitmap{
		.bytes = std::span{static_cast<std::byte const*>(result), std::size_t(size.x * size.y * int(channels_v))},
		.size = size,
	};

	return true;
}
} // namespace kvf

// color_bitmap

#include <kvf/color_bitmap.hpp>

namespace kvf {
auto Color::to_srgb() const -> glm::vec4 { return glm::convertLinearToSRGB(to_vec4()); }
auto Color::to_linear() const -> glm::vec4 { return glm::convertSRGBToLinear(to_vec4()); }

void ColorBitmap::resize(glm::ivec2 size) {
	if (size.x < 0 || size.y < 0) { return; }
	m_size = size;
	m_bitmap.resize(std::size_t(m_size.x * m_size.y));
}

auto ColorBitmap::at(int const x, int const y) const -> Color const& {
	auto const index = (y * m_size.x) + x;
	return m_bitmap.at(std::size_t(index));
}

auto ColorBitmap::at(int const x, int const y) -> Color& {
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
	return const_cast<Color&>(std::as_const(*this).at(x, y));
}

auto ColorBitmap::bitmap() const -> Bitmap {
	static_assert(sizeof(Color) == Bitmap::channels_v);
	void const* first = m_bitmap.data();
	return Bitmap{
		.bytes = std::span{static_cast<std::byte const*>(first), sizeof(Color) * m_bitmap.size()},
		.size = m_size,
	};
}
} // namespace kvf

// util

namespace kvf {
namespace {
template <typename T>
auto read_from_file(T& out, klib::CString path) -> IoResult {
	using value_type = T::value_type;
	auto file = std::ifstream{path.c_str(), std::ios::binary | std::ios::ate};
	if (!file.is_open()) { return IoResult::OpenFailed; }
	auto const size = file.tellg();
	if (std::size_t(size) % sizeof(value_type) != 0) { return IoResult::SizeMismatch; }
	file.seekg(0, std::ios::beg);
	out.resize(std::size_t(size) / sizeof(value_type));
	void* first = out.data();
	file.read(static_cast<char*>(first), size);
	return IoResult::Success;
}

struct MakeMipMaps {
	// NOLINTNEXTLINE
	vma::Image& out;

	vk::CommandBuffer command_buffer;

	vk::ImageMemoryBarrier2 barrier{};
	vk::ImageAspectFlags aspect{};
	std::uint32_t layer_count{};

	auto blit_mips(std::uint32_t const src_level, vk::Offset3D const src_offset, vk::Offset3D const dst_offset) const -> void {
		auto ib = vk::ImageBlit2{};
		ib.srcSubresource.setAspectMask(aspect).setMipLevel(src_level).setLayerCount(layer_count);
		ib.dstSubresource.setAspectMask(aspect).setMipLevel(src_level + 1).setLayerCount(layer_count);
		ib.srcOffsets[1] = src_offset;
		ib.dstOffsets[1] = dst_offset;
		auto bii = vk::BlitImageInfo2{};
		bii.setSrcImage(barrier.image)
			.setDstImage(barrier.image)
			.setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
			.setRegions(ib)
			.setFilter(vk::Filter::eLinear);
		command_buffer.blitImage2(bii);
	}

	auto blit_next_mip(std::uint32_t const src_level, vk::Offset3D const src_offset, vk::Offset3D const dst_offset) -> void {
		barrier.subresourceRange.setBaseMipLevel(src_level + 1);
		barrier.setOldLayout(vk::ImageLayout::eUndefined).setNewLayout(vk::ImageLayout::eTransferDstOptimal);
		util::record_barrier(command_buffer, barrier);

		blit_mips(src_level, src_offset, dst_offset);

		barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal).setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
		util::record_barrier(command_buffer, barrier);
	}

	auto operator()() -> void {
		barrier = out.get_render_device()->image_barrier(out.get_info().aspect);
		layer_count = out.get_info().layers;
		aspect = out.get_info().aspect;

		barrier.setImage(out.get_image())
			.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
			.setDstAccessMask(barrier.srcAccessMask)
			.setDstStageMask(barrier.srcStageMask)
			.setOldLayout(out.get_layout())
			.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
		barrier.subresourceRange.setAspectMask(aspect).setLevelCount(1).setLayerCount(layer_count);
		util::record_barrier(command_buffer, barrier);

		auto src_extent = vk::Extent3D{out.get_extent(), 1};
		for (std::uint32_t mip = 0; mip + 1 < out.get_mip_levels(); ++mip) {
			vk::Extent3D dst_extent = vk::Extent3D(std::max(src_extent.width / 2, 1u), std::max(src_extent.height / 2, 1u), 1u);
			auto const src_offset = vk::Offset3D{static_cast<int>(src_extent.width), static_cast<int>(src_extent.height), 1};
			auto const dst_offset = vk::Offset3D{static_cast<int>(dst_extent.width), static_cast<int>(dst_extent.height), 1};
			blit_next_mip(mip, src_offset, dst_offset);
			src_extent = dst_extent;
		}
	}
};
} // namespace

auto util::color_from_hex(std::string_view hex) -> Color {
	if (hex.size() != 9 || !hex.starts_with('#')) { return {}; }
	hex = hex.substr(1);
	auto const next = [&](std::uint8_t& out) {
		auto const [ptr, ec] = std::from_chars(hex.data(), hex.data() + 2, out, 16);
		hex = hex.substr(2);
		return ec == std::errc{} && ptr == hex.data();
	};
	auto ret = Color{};
	if (!next(ret.x) || !next(ret.y) || !next(ret.z) || !next(ret.w)) { return {}; }
	return ret;
}

auto util::to_hex_string(Color const& color) -> std::string { return std::format("#{:02x}{:02x}{:02x}{:02x}", color.x, color.y, color.z, color.w); }

auto util::compute_mip_levels(vk::Extent2D const extent) -> std::uint32_t {
	return static_cast<std::uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1u;
}

auto util::wait_for_fence(vk::Device device, vk::Fence fence, std::chrono::nanoseconds const timeout) -> bool {
	return device.waitForFences(fence, vk::True, std::uint64_t(timeout.count())) == vk::Result::eSuccess;
}

void util::record_barriers(vk::CommandBuffer const command_buffer, std::span<vk::ImageMemoryBarrier2 const> image_barriers) {
	auto di = vk::DependencyInfo{};
	di.pImageMemoryBarriers = image_barriers.data();
	di.imageMemoryBarrierCount = static_cast<std::uint32_t>(image_barriers.size());
	command_buffer.pipelineBarrier2(di);
}

auto util::string_from_file(std::string& out_string, klib::CString path) -> IoResult { return read_from_file(out_string, path); }
auto util::bytes_from_file(std::vector<std::byte>& out_bytes, klib::CString path) -> IoResult { return read_from_file(out_bytes, path); }
auto util::spirv_from_file(std::vector<std::uint32_t>& out_code, klib::CString path) -> IoResult { return read_from_file(out_code, path); }

auto util::overwrite(vma::Buffer& dst, BufferWrite const bytes, vk::DeviceSize offset) -> bool {
	if (!dst) { return false; }

	if (dst.get_size() < offset + bytes.size()) { return false; }
	if (bytes.is_empty()) { return true; }

	if (auto* ptr = static_cast<std::byte*>(dst.get_mapped())) {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		std::memcpy(ptr + offset, bytes.data(), bytes.size());
		return true;
	}

	if ((dst.get_info().usage & vk::BufferUsageFlagBits::eTransferDst) != vk::BufferUsageFlagBits::eTransferDst) { return false; }

	auto const bci = vma::BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eTransferSrc,
		.type = vma::BufferType::Host,
	};
	auto staging = vma::Buffer{dst.get_render_device(), bci, bytes.size()};
	if (!overwrite(staging, bytes)) { return false; }

	auto const bc = vk::BufferCopy2{0, offset, staging.get_size()};
	auto cbi = vk::CopyBufferInfo2{};
	cbi.setSrcBuffer(staging.get_buffer()).setDstBuffer(dst.get_buffer()).setRegions(bc);
	auto cmd = CommandBuffer{dst.get_render_device()};
	cmd.get().copyBuffer2(cbi);
	return cmd.submit_and_wait();
}

auto util::write_to(vma::Buffer& dst, BufferWrite const bytes) -> bool {
	if (!dst.resize(bytes.size())) { return false; }
	return overwrite(dst, bytes);
}

auto util::write_to(vma::Image& dst, std::span<Bitmap const> layers) -> bool {
	if (!dst || dst.get_info().layers != layers.size()) { return false; }
	if ((dst.get_info().usage & vk::ImageUsageFlagBits::eTransferDst) != vk::ImageUsageFlagBits::eTransferDst) { return false; }
	auto const size = layers.front().size;
	auto const layer_size = vk::DeviceSize(size.x * size.y * std::int32_t(Bitmap::channels_v));
	auto const total_size = layers.size() * layer_size;
	auto const check = [size, layer_size](Bitmap const& b) { return b.size == size && b.bytes.size() == layer_size; };
	if (!std::ranges::all_of(layers, check)) { return false; }

	if (layer_size == 0) { return true; }
	auto const extent = to_vk_extent(size);
	if (!dst.resize(extent)) { return false; }

	auto const original_layout = dst.get_layout();

	auto const bci = vma::BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eTransferSrc,
		.type = vma::BufferType::Host,
	};
	auto staging = vma::Buffer{dst.get_render_device(), bci, total_size};

	auto cmd = CommandBuffer{dst.get_render_device()};
	auto barrier = vk::ImageMemoryBarrier2{};
	barrier.setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setOldLayout(vk::ImageLayout::eUndefined)
		.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
	dst.transition(cmd, barrier);

	auto span = std::span{static_cast<std::byte*>(staging.get_mapped()), total_size};
	auto buffer_offset = vk::DeviceSize{};
	auto cbtii = vk::CopyBufferToImageInfo2{};
	cbtii.setDstImage(dst.get_image()).setDstImageLayout(vk::ImageLayout::eTransferDstOptimal).setSrcBuffer(staging.get_buffer());
	for (auto const [index, layer] : std::ranges::enumerate_view(layers)) {
		std::memcpy(span.data(), layer.bytes.data(), layer_size);

		auto bic = vk::BufferImageCopy2{};
		bic.setImageExtent({extent.width, extent.height, 1})
			.setImageSubresource(vk::ImageSubresourceLayers{dst.get_info().aspect, 0, std::uint32_t(index), 1})
			.setBufferOffset(buffer_offset);
		cbtii.setRegions(bic);
		cmd.get().copyBufferToImage2(cbtii);

		span = span.subspan(layer_size);
		buffer_offset += layer_size;
	}

	auto current_layout = dst.get_layout();
	if (dst.get_mip_levels() > 1) {
		MakeMipMaps{.out = dst, .command_buffer = cmd}();
		current_layout = vk::ImageLayout::eTransferSrcOptimal;
	}

	auto const final_layout = original_layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eShaderReadOnlyOptimal : original_layout;
	barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite)
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
		.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
		.setOldLayout(current_layout)
		.setNewLayout(final_layout);
	dst.transition(cmd, barrier);

	return cmd.submit_and_wait();
}
} // namespace kvf
