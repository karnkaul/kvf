#include <vk_mem_alloc.h>
#include <klib/unique.hpp>
#include <klib/version_str.hpp>
#include <kvf/error.hpp>
#include <log.hpp>
#include <vulkan/vulkan.hpp>
#include <chrono>

// render_device

#include <kvf/render_device.hpp>

namespace kvf {
namespace {
namespace chr = std::chrono;
using namespace std::chrono_literals;

constexpr auto srgb_formats_v = std::array{vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb, vk::Format::eA8B8G8R8SrgbPack32};

constexpr auto to_str(vk::PresentModeKHR const present_mode) -> std::string_view {
	switch (present_mode) {
	case vk::PresentModeKHR::eFifo: return "FIFO";
	case vk::PresentModeKHR::eFifoRelaxed: return "FIFO Relaxed";
	case vk::PresentModeKHR::eMailbox: return "Mailbox";
	case vk::PresentModeKHR::eImmediate: return "Immediate";
	default: return "Unsupported";
	}
}

struct GpuImpl {
	Gpu gpu{};
	std::uint32_t queue_family{};

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
		auto ret = std::vector<GpuImpl>{};
		auto const get_queue_family = [](vk::PhysicalDevice device, std::uint32_t& out) {
			for (auto const& [index, family] : std::ranges::enumerate_view(device.getQueueFamilyProperties())) {
				if ((family.queueFlags & queue_flags_v) != queue_flags_v) { continue; }
				out = static_cast<std::uint32_t>(index);
				return true;
			}
			return false;
		};
		for (auto const& device : all_devices) {
			auto gpu = GpuImpl{.gpu = Gpu{.device = device, .properties = device.getProperties(), .features = device.getFeatures()}};
			if (device.getProperties().apiVersion < VK_API_VERSION_1_3) { continue; }
			if (!has_required_extensions(device.enumerateDeviceExtensionProperties())) { continue; }
			if (!get_queue_family(device, gpu.queue_family)) { continue; }
			if (device.getSurfaceSupportKHR(gpu.queue_family, surface) == vk::False) { continue; }
			ret.push_back(gpu);
		}
		return ret;
	}
};

struct DeviceBlockDeleter {
	void operator()(vk::Device device) const { device.waitIdle(); }
};
using DeviceBlock = klib::Unique<vk::Device, DeviceBlockDeleter>;

[[nodiscard]] constexpr auto get_optimal_present_mode(std::span<vk::PresentModeKHR const> present_modes) {
	static constexpr auto desired_v = std::array{vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eFifoRelaxed};
	for (auto const desired : desired_v) {
		if (std::ranges::find(present_modes, desired) != present_modes.end()) { return desired; }
	}
	return vk::PresentModeKHR::eFifo;
}

[[nodiscard]] constexpr auto get_surface_format(std::span<vk::SurfaceFormatKHR const> supported) -> vk::SurfaceFormatKHR {
	for (auto const srgb_format : srgb_formats_v) {
		auto const it = std::ranges::find_if(supported, [srgb_format](vk::SurfaceFormatKHR const& format) {
			return format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear && format.format == srgb_format;
		});
		if (it != supported.end()) { return *it; }
	}
	return vk::SurfaceFormatKHR{};
}

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

void record_barriers(vk::CommandBuffer const command_buffer, std::span<vk::ImageMemoryBarrier2 const> image_barriers) {
	auto di = vk::DependencyInfo{};
	di.pImageMemoryBarriers = image_barriers.data();
	di.imageMemoryBarrierCount = static_cast<std::uint32_t>(image_barriers.size());
	command_buffer.pipelineBarrier2(di);
}

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
		log::info("Swapchain extent: {}x{}, images: {}, mode: {}", extent.width, extent.height, m_images.size(), to_str(m_info.presentMode));

		// TODO: remove
		auto barrier = vk::ImageMemoryBarrier2{};
		barrier.setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead)
			.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead)
			.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
			.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::ePresentSrcKHR)
			.setSrcQueueFamilyIndex(*m_info.pQueueFamilyIndices)
			.setDstQueueFamilyIndex(*m_info.pQueueFamilyIndices)
			.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
		auto barriers = std::vector<vk::ImageMemoryBarrier2>{};
		barriers.reserve(m_images.size());
		for (auto const image : m_images) { barriers.emplace_back(barrier).setImage(image); }

		auto pool = m_device.createCommandPoolUnique(vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, *m_info.pQueueFamilyIndices});
		auto cmd = m_device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{*pool, vk::CommandBufferLevel::ePrimary, 1}).front();
		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		record_barriers(cmd, barriers);
		cmd.end();

		auto const cbsi = vk::CommandBufferSubmitInfo{cmd};
		auto si = vk::SubmitInfo2{};
		si.setCommandBufferInfos(cbsi);
		auto fence = m_device.createFenceUnique({});
		m_queue.submit2(si, *fence);
		[[maybe_unused]] auto const result = m_device.waitForFences(*fence, vk::True, chr::nanoseconds{2s}.count());
	}

	[[nodiscard]] auto get_image_index() const -> std::optional<std::uint32_t> { return m_image_index; }

	auto acquire_next_image(vk::Extent2D const framebuffer, vk::Semaphore const ssignal) -> bool {
		if (m_image_index) { return true; }

		static constexpr auto timeout_v = chr::nanoseconds{5s};

		auto image_index = std::uint32_t{};
		auto const result = m_device.acquireNextImageKHR(*m_swapchain, timeout_v.count(), ssignal, {}, &image_index);
		switch (result) {
		case vk::Result::eErrorOutOfDateKHR:
		case vk::Result::eSuboptimalKHR: recreate(framebuffer); return false;
		case vk::Result::eSuccess: m_image_index = image_index; return true;
		default: throw Error{"Failed to acquire next Swapchain Image"};
		}
	}

	void present(vk::Queue queue, vk::Extent2D const framebuffer, vk::Semaphore const wait) {
		if (!m_image_index) { return; }

		auto pi = vk::PresentInfoKHR{};
		pi.setImageIndices(*m_image_index).setSwapchains(*m_swapchain).setWaitSemaphores(wait);
		auto const result = queue.presentKHR(&pi);
		switch (result) {
		case vk::Result::eErrorOutOfDateKHR:
		case vk::Result::eSuboptimalKHR: recreate(framebuffer); break;
		default: break;
		}
		m_image_index.reset();
	}

	[[nodiscard]] auto get_info() const -> vk::SwapchainCreateInfoKHR const& { return m_info; }
	[[nodiscard]] auto get_images() const -> std::span<vk::Image const> { return m_images; }
	[[nodiscard]] auto get_image_views() const -> std::span<vk::UniqueImageView const> { return m_image_views; }

  private:
	[[nodiscard]] static constexpr auto get_image_extent(vk::SurfaceCapabilitiesKHR const& caps, vk::Extent2D framebuffer) -> vk::Extent2D {
		static constexpr auto limitless_v = std::numeric_limits<std::uint32_t>::max();
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
} // namespace

struct RenderDevice::Impl {
	Impl(IWsi const* wsi, Flags const flags) : m_wsi(wsi) {
		auto const validation = (flags & RenderDeviceFlag::ValidationLayers) == RenderDeviceFlag::ValidationLayers;
		create_instance(validation);
		create_surface();
		select_gpu();
		create_device();
		create_swapchain();
	}

	[[nodiscard]] auto get_loader_api_version() const -> klib::Version { return m_loader_version; }
	[[nodiscard]] auto get_instance() const -> vk::Instance { return *m_instance; }
	[[nodiscard]] auto get_surface() const -> vk::SurfaceKHR { return *m_surface; }
	[[nodiscard]] auto get_gpu() const -> Gpu const& { return m_gpu.gpu; }
	[[nodiscard]] auto get_device() const -> vk::Device { return *m_device; }
	[[nodiscard]] auto get_queue_family() const -> std::uint32_t { return m_gpu.queue_family; }

	[[nodiscard]] auto get_present_mode() const -> vk::PresentModeKHR { return m_swapchain.get_info().presentMode; }
	[[nodiscard]] auto get_supported_present_modes() const -> std::span<vk::PresentModeKHR const> { return m_present_modes; }

	auto request_present_mode(vk::PresentModeKHR desired) -> bool {
		if (std::ranges::find(m_present_modes, desired) == m_present_modes.end()) { return false; }
		m_next_present_mode = desired;
		return true;
	}

	auto acquire_next_image() -> bool {
		auto const framebuffer_extent = m_wsi->get_framebuffer_size();
		if (m_next_present_mode || m_swapchain.get_info().imageExtent != framebuffer_extent) {
			auto const present_mode = m_next_present_mode.value_or(m_swapchain.get_info().presentMode);
			m_swapchain.recreate(m_wsi->get_framebuffer_size(), present_mode);
			m_next_present_mode.reset();
		}

		auto const& sync = m_syncs.at(m_frame_index);

		static constexpr auto timeout_v = chr::nanoseconds{5s};
		if (m_device->waitForFences(*sync.drawn, vk::True, timeout_v.count()) != vk::Result::eSuccess) { throw Error{"Failed to wait for Render Fence"}; }
		m_device->resetFences(*sync.drawn);

		auto lock = std::scoped_lock{m_queue_mutex};
		return m_swapchain.acquire_next_image(m_wsi->get_framebuffer_size(), *sync.draw);
	}

	void temp_render() {
		auto const image_index = m_swapchain.get_image_index();
		if (!image_index) { return; }

		auto const image_view = *m_swapchain.get_image_views()[*image_index];

		auto render_area = vk::Rect2D{};
		render_area.setExtent(m_swapchain.get_info().imageExtent);

		auto cai = vk::RenderingAttachmentInfo{};
		cai.setImageView(image_view)
			.setImageLayout(vk::ImageLayout::eAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearColorValue{std::array{1.0f, 0.0f, 0.0f, 1.0f}});
		auto ri = vk::RenderingInfo{};
		ri.setColorAttachments(cai).setLayerCount(1).setRenderArea(render_area);

		auto& cmd = m_command_buffers.at(m_frame_index);
		cmd.begin(vk::CommandBufferBeginInfo{});
		// TODO: layout transitions
		cmd.beginRendering(ri);
		cmd.endRendering();
		// TODO: layout transitions
		cmd.end();

		auto const& sync = m_syncs.at(m_frame_index);
		auto const cbsi = vk::CommandBufferSubmitInfo{cmd};
		auto const wssi = vk::SemaphoreSubmitInfo{*sync.draw, 0, vk::PipelineStageFlagBits2::eTopOfPipe};
		auto const sssi = vk::SemaphoreSubmitInfo{*sync.present, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput};
		auto si = vk::SubmitInfo2{};
		si.setCommandBufferInfos(cbsi).setWaitSemaphoreInfos(wssi).setSignalSemaphoreInfos(sssi);

		auto lock = std::scoped_lock{m_queue_mutex};
		m_queue.submit2(si, *sync.drawn);
	}

	void present_acquired_image() {
		auto lock = std::scoped_lock{m_queue_mutex};
		auto const& sync = m_syncs.at(m_frame_index);
		m_swapchain.present(m_queue, m_wsi->get_framebuffer_size(), *sync.present);
		m_frame_index = (m_frame_index + 1) % buffering_v;
	}

  private:
	struct Sync {
		vk::UniqueSemaphore draw{};
		vk::UniqueSemaphore present{};
		vk::UniqueFence drawn{};
	};

	void create_instance(bool validation) {
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
		auto const wsi_extensions = m_wsi->get_instance_extensions();
		auto extensions = std::vector(wsi_extensions.begin(), wsi_extensions.end());
		if (validation) {
			static constexpr char const* validation_layer_v = "VK_LAYER_KHRONOS_validation";
			auto const props = vk::enumerateInstanceLayerProperties();
			static constexpr auto pred = [](vk::LayerProperties const& p) { return p.layerName == std::string_view{validation_layer_v}; };
			auto const it = std::ranges::find_if(props, pred);
			if (it == props.end()) {
				log::warn("Validation layers requested but {} is not available", validation_layer_v);
				validation = false;
			} else {
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
				ici.setPEnabledLayerNames(validation_layer_v);
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

		if (validation) { create_debug_messenger(); }
	}

	void create_debug_messenger() {
		auto dmci = vk::DebugUtilsMessengerCreateInfoEXT{};
		using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
		using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
		using Types = vk::DebugUtilsMessageTypeFlagsEXT;
		using Data = vk::DebugUtilsMessengerCallbackDataEXT;
		static constexpr auto severity_v = Severity::eError | Severity::eWarning;
		static constexpr auto types_v = Type::eGeneral | Type::ePerformance | Type::eValidation | Type::eDeviceAddressBinding;
		static auto const on_msg = [](Severity severity, Types /*unused*/, Data const* data, void* /*unused*/) -> vk::Bool32 {
			static constexpr std::string_view tag_v{"validation"};
			switch (severity) {
			case Severity::eError: klib::log::error(tag_v, "{}", data->pMessage); break;
			case Severity::eWarning: klib::log::warn(tag_v, "{}", data->pMessage); break;
			default: break;
			}
			return vk::False;
		};
		dmci.setMessageSeverity(severity_v).setMessageType(types_v).setPfnUserCallback(on_msg);
		m_debug_messenger = m_instance->createDebugUtilsMessengerEXTUnique(dmci);
		log::debug("Vulkan Debug Messenger created");
	}

	void create_surface() {
		auto surface = m_wsi->create_surface(*m_instance);
		if (!surface) { throw Error{"Failed to create Vulkan Surface"}; }
		m_surface = vk::UniqueSurfaceKHR{surface, *m_instance};
	}

	void select_gpu() {
		auto const devices = m_instance->enumeratePhysicalDevices();
		auto gpus = GpuImpl::get_viable(*m_instance, *m_surface);
		if (gpus.empty()) { throw Error{"No viable GPUs"}; }
		std::ranges::sort(gpus, [this](GpuImpl const& a, GpuImpl const& b) { return m_wsi->compare_gpus(a.gpu, b.gpu); });
		m_gpu = gpus.front();
		log::debug("Using GPU: {}, queue family: {}", m_gpu.gpu.properties.deviceName.data(), m_gpu.queue_family);
	}

	void create_device() {
		auto qci = vk::DeviceQueueCreateInfo{};
		static constexpr auto queue_priorities_v = std::array{1.0f};
		qci.setQueueFamilyIndex(m_gpu.queue_family).setQueueCount(1).setQueuePriorities(queue_priorities_v);

		auto enabled_features = vk::PhysicalDeviceFeatures{};
		enabled_features.fillModeNonSolid = m_gpu.gpu.features.fillModeNonSolid;
		enabled_features.wideLines = m_gpu.gpu.features.wideLines;
		enabled_features.samplerAnisotropy = m_gpu.gpu.features.samplerAnisotropy;
		enabled_features.sampleRateShading = m_gpu.gpu.features.sampleRateShading;

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

		m_device = m_gpu.gpu.device.createDeviceUnique(dci);
		if (!m_device) { throw Error{"Failed to create Vulkan Device"}; }
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_device);

		m_queue = m_device->getQueue(m_gpu.queue_family, 0);
		log::debug("Vulkan Device created");

		m_device_block.get() = *m_device;
	}

	void create_swapchain() {
		auto const surface_format = get_surface_format(m_gpu.gpu.device.getSurfaceFormatsKHR(*m_surface));
		m_present_modes = m_gpu.gpu.device.getSurfacePresentModesKHR(*m_surface);
		auto sci = vk::SwapchainCreateInfoKHR{};
		sci.surface = *m_surface;
		sci.presentMode = get_optimal_present_mode(m_present_modes);
		sci.imageFormat = surface_format.format;
		sci.queueFamilyIndexCount = 1u;
		sci.pQueueFamilyIndices = &m_gpu.queue_family;
		sci.imageColorSpace = surface_format.colorSpace;
		m_swapchain.init(*m_device, m_gpu.gpu.device, sci, m_queue);
		m_swapchain.recreate(m_wsi->get_framebuffer_size());

		auto const cpci = vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_gpu.queue_family};
		m_command_pool = m_device->createCommandPoolUnique(cpci);
		auto const cbai = vk::CommandBufferAllocateInfo{*m_command_pool, vk::CommandBufferLevel::ePrimary, std::uint32_t(buffering_v)};
		if (m_device->allocateCommandBuffers(&cbai, m_command_buffers.data()) != vk::Result::eSuccess) {
			throw Error{"Failed to allocate render CommandBuffer(s)"};
		}

		for (auto& sync : m_syncs) {
			sync.draw = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
			sync.present = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
			sync.drawn = m_device->createFenceUnique(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
		}
	}

	IWsi const* m_wsi{};

	klib::Version m_loader_version{};

	vk::UniqueInstance m_instance{};
	vk::UniqueDebugUtilsMessengerEXT m_debug_messenger{};
	vk::UniqueSurfaceKHR m_surface{};
	GpuImpl m_gpu{};

	vk::UniqueDevice m_device{};
	vk::Queue m_queue{};
	std::mutex m_queue_mutex{};

	std::vector<vk::PresentModeKHR> m_present_modes{};
	Swapchain m_swapchain{};
	Buffered<Sync> m_syncs{};
	vk::UniqueCommandPool m_command_pool{};
	Buffered<vk::CommandBuffer> m_command_buffers{};
	std::size_t m_frame_index{};

	std::optional<vk::PresentModeKHR> m_next_present_mode{};

	DeviceBlock m_device_block{};
};

void RenderDevice::Deleter::operator()(Impl* ptr) const noexcept { std::default_delete<Impl>{}(ptr); }

RenderDevice::RenderDevice(gsl::not_null<IWsi const*> wsi, Flags const flags) : m_impl(new Impl(wsi, flags)) {}

auto RenderDevice::get_loader_api_version() const -> klib::Version { return m_impl->get_loader_api_version(); }
auto RenderDevice::get_instance() const -> vk::Instance { return m_impl->get_instance(); }
auto RenderDevice::get_surface() const -> vk::SurfaceKHR { return m_impl->get_surface(); }
auto RenderDevice::get_gpu() const -> Gpu const& { return m_impl->get_gpu(); }
auto RenderDevice::get_device() const -> vk::Device { return m_impl->get_device(); }
auto RenderDevice::get_queue_family() const -> std::uint32_t { return m_impl->get_queue_family(); }

auto RenderDevice::get_present_mode() const -> vk::PresentModeKHR { return m_impl->get_present_mode(); }
auto RenderDevice::get_supported_present_modes() const -> std::span<vk::PresentModeKHR const> { return m_impl->get_supported_present_modes(); }
auto RenderDevice::request_present_mode(vk::PresentModeKHR const desired) -> bool { return m_impl->request_present_mode(desired); }

auto RenderDevice::acquire_next_image() -> bool { return m_impl->acquire_next_image(); }
void RenderDevice::present_acquired_image() { m_impl->present_acquired_image(); }
void RenderDevice::temp_render() { m_impl->temp_render(); }
} // namespace kvf
