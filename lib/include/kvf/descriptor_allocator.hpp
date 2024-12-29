#pragma once
#include <klib/polymorphic.hpp>
#include <vulkan/vulkan.hpp>
#include <span>

namespace kvf {
class IDescriptorAllocator : public klib::Polymorphic {
  public:
	virtual auto allocate(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> layouts) -> bool = 0;
};
} // namespace kvf
