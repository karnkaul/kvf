#include "kvf/scratch_command_buffer.hpp"
#include "kvf/panic.hpp"
#include "kvf/util.hpp"

namespace kvf {
ScratchCommandBuffer::ScratchCommandBuffer(gsl::not_null<IRenderDevice*> render_device) : m_render_device(render_device) {
	auto cpci = vk::CommandPoolCreateInfo{};
	cpci.setQueueFamilyIndex(m_render_device->get_queue_family()).setFlags(vk::CommandPoolCreateFlagBits::eTransient);
	m_pool = m_render_device->get_device().createCommandPoolUnique(cpci);
	auto cbai = vk::CommandBufferAllocateInfo{};
	cbai.setCommandPool(*m_pool).setCommandBufferCount(1);
	if (m_render_device->get_device().allocateCommandBuffers(&cbai, &m_cmd) != vk::Result::eSuccess) {
		throw Panic{"Failed to allocate Vulkan Command Buffer"};
	}
	m_cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
}

auto ScratchCommandBuffer::submit_and_wait(std::chrono::seconds const timeout) -> bool {
	m_cmd.end();
	auto const cbsi = vk::CommandBufferSubmitInfo{m_cmd};
	auto si = vk::SubmitInfo2{};
	si.setCommandBufferInfos(cbsi);
	auto const fence = m_render_device->get_device().createFenceUnique({});
	m_render_device->queue_submit(si, *fence);
	return util::wait_for_fence(m_render_device->get_device(), *fence, timeout);
}
} // namespace kvf
