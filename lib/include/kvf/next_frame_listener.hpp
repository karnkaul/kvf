#pragma once
#include "klib/base_types.hpp"
#include "kvf/frame_index.hpp"

namespace kvf {
class INextFrameListener : public klib::Polymorphic {
  public:
	virtual void on_next_frame(FrameIndex frame_index) = 0;
};
} // namespace kvf
