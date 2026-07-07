#pragma once
#include "klib/base_types.hpp"
#include <vulkan/vulkan.hpp>
#include <span>

namespace kvf {
class IRingDescriptorAllocator : public klib::Polymorphic {
  public:
	[[nodiscard]] virtual auto allocate_next(std::span<vk::DescriptorSet> out_sets, std::span<vk::DescriptorSetLayout const> set_layouts) -> bool = 0;
};
} // namespace kvf
