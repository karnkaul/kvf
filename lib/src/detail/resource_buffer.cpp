#include "detail/resource_buffer.hpp"
#include "klib/debug/assert.hpp"
#include "kvf/panic.hpp"
#include "kvf/scratch_command_buffer.hpp"
#include "kvf/util.hpp"
#include <numeric>

namespace kvf {
namespace detail {
ResourceBuffer::ResourceBuffer(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info) : m_render_device(render_device) {
	recreate_impl(create_info);
}

void ResourceBuffer::resize(vk::DeviceSize size) {
	util::ensure_positive(size);

	if (m_buffer && m_capacity >= size) {
		m_size = size;
		return;
	}

	recreate_impl(CreateInfo{.usage = m_usage, .type = m_type, .size = size});
}

auto ResourceBuffer::write_contiguous(std::span<BufferWrite const> writes, vk::DeviceSize const write_size, vk::DeviceSize const offset) -> bool {
	if (get_size() < offset + write_size) { return false; }
	if (write_size == 0) { return true; }

	if (auto dst = get_mapped_span(); !dst.empty()) {
		KLIB_ASSERT(dst.size() >= offset + write_size);
		dst = dst.subspan(offset);
		for (auto const write : writes) {
			if (write.is_empty()) { continue; }
			std::memcpy(dst.data(), write.data(), write.size());
			dst = dst.subspan(write.size());
		}
		return true;
	}

	if ((m_usage & vk::BufferUsageFlagBits::eTransferDst) != vk::BufferUsageFlagBits::eTransferDst) { return false; }

	auto const bci = BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eTransferSrc,
		.type = BufferType::Host,
		.size = write_size,
	};
	auto staging = ResourceBuffer{m_render_device, bci};
	if (!staging.write_contiguous(writes, write_size, 0)) { return false; }

	auto const bc = vk::BufferCopy2{0, offset, staging.get_size()};
	auto cbi = vk::CopyBufferInfo2{};
	cbi.setSrcBuffer(staging.get_buffer()).setDstBuffer(get_buffer()).setRegions(bc);

	auto cmd = ScratchCommandBuffer{m_render_device};
	cmd.get().copyBuffer2(cbi);
	return cmd.submit_and_wait();
}

void ResourceBuffer::recreate_impl(CreateInfo info) {
	if (info.type == BufferType::Device) { info.usage |= vk::BufferUsageFlagBits::eTransferDst; }
	util::ensure_positive(info.size);

	auto allocation_ci = VmaAllocationCreateInfo{};
	allocation_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	if (info.type == BufferType::Device) {
		allocation_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	} else {
		allocation_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocation_ci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
	}

	auto const buffer_ci = vk::BufferCreateInfo{{}, info.size, info.usage};
	auto c_buffer_ci = static_cast<VkBufferCreateInfo>(buffer_ci);

	VmaAllocation allocation{};
	VkBuffer buffer{};
	auto alloc_info = VmaAllocationInfo{};
	if (vmaCreateBuffer(m_render_device->get_allocator(), &c_buffer_ci, &allocation_ci, &buffer, &allocation, &alloc_info) != VK_SUCCESS) {
		throw Panic{"Failed to create Vulkan Buffer"};
	}

	destroy();

	m_usage = info.usage;
	m_type = info.type;
	m_capacity = m_size = info.size;
	m_buffer = buffer;
	m_allocation = allocation;
	m_mapped = alloc_info.pMappedData;
}

void ResourceBuffer::destroy() {
	if (!m_buffer) { return; }
	vmaDestroyBuffer(m_render_device->get_allocator(), m_buffer, m_allocation);
	m_size = m_capacity = 0;
	m_buffer = vk::Buffer{};
	m_allocation = {};
	m_mapped = nullptr;
}
} // namespace detail

auto IBuffer::write_in_place(BufferWrite const write, vk::DeviceSize const offset) -> bool { return write_contiguous({&write, 1}, write.size(), offset); }

void IBuffer::resize_overwrite_contiguous(std::span<BufferWrite const> writes) {
	auto const total_size = std::accumulate(writes.begin(), writes.end(), 0uz, [](std::size_t i, BufferWrite const& w) { return i + w.size(); });
	resize(total_size);
	write_contiguous(writes, total_size, 0);
}

auto IBuffer::get_mapped_span() const -> std::span<std::byte> {
	auto* bytes = get_mapped_ptr();
	if (bytes == nullptr) { return {}; }
	return {static_cast<std::byte*>(bytes), get_size()};
}

auto IBuffer::descriptor_info() const -> vk::DescriptorBufferInfo {
	auto ret = vk::DescriptorBufferInfo{};
	ret.setBuffer(get_buffer()).setRange(get_size());
	return ret;
}
} // namespace kvf
