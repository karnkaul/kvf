#pragma once
#include <klib/unique.hpp>
#include <vulkan/vulkan.hpp>

namespace kvf {
struct DeviceBlockDeleter {
	void operator()(vk::Device device) const { device.waitIdle(); }
};
using DeviceBlock = klib::Unique<vk::Device, DeviceBlockDeleter>;
} // namespace kvf
