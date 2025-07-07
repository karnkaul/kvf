#pragma once
#include <klib/unique.hpp>
#include <vulkan/vulkan.hpp>

namespace kvf {
struct DeviceWaiterDeleter {
	void operator()(vk::Device device) const { device.waitIdle(); }
};
using DeviceWaiter = klib::Unique<vk::Device, DeviceWaiterDeleter>;
} // namespace kvf
