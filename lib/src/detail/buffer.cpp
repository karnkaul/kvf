#include "detail/buffer.hpp"
#include "klib/debug/assert.hpp"
#include "kvf/render_device.hpp"
#include "kvf/scratch_command_buffer.hpp"
#include "kvf/util.hpp"
#include <numeric>

namespace kvf {
namespace detail {
Buffer::Buffer(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info) : m_render_device(render_device) { recreate_impl(create_info); }

void Buffer::resize(vk::DeviceSize size) {
	util::ensure_positive(size);

	if (m_buffer.get().buffer && m_info.size >= size) {
		m_size = size;
		return;
	}

	auto info = m_info;
	info.size = size;
	recreate_impl(info);
}

auto Buffer::write_contiguous(std::span<BufferWrite const> writes, vk::DeviceSize const write_size, vk::DeviceSize const offset) -> bool {
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

	if ((m_info.usage & vk::BufferUsageFlagBits::eTransferDst) != vk::BufferUsageFlagBits::eTransferDst) { return false; }

	auto const bci = BufferCreateInfo{
		.usage = vk::BufferUsageFlagBits::eTransferSrc,
		.type = BufferType::Host,
		.size = write_size,
	};
	auto staging = Buffer{m_render_device, bci};
	if (!staging.write_contiguous(writes, write_size, 0)) { return false; }

	auto const bc = vk::BufferCopy2{0, offset, staging.get_size()};
	auto cbi = vk::CopyBufferInfo2{};
	cbi.setSrcBuffer(staging.get_buffer()).setDstBuffer(get_buffer()).setRegions(bc);

	auto cmd = ScratchCommandBuffer{m_render_device};
	cmd.get().copyBuffer2(cbi);
	return cmd.submit_and_wait();
}

void Buffer::recreate_impl(CreateInfo create_info) {
	if (create_info.type == BufferType::Device) { create_info.usage |= vk::BufferUsageFlagBits::eTransferDst; }
	util::ensure_positive(create_info.size);

	m_buffer = vma::create_buffer(m_render_device->get_allocator(), create_info);
	m_info = create_info;
	m_size = create_info.size;
}
} // namespace detail

auto IBuffer::create(gsl::not_null<IRenderDevice*> render_device, CreateInfo const& create_info) -> std::unique_ptr<IBuffer> {
	return std::make_unique<detail::Buffer>(render_device, create_info);
}

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
