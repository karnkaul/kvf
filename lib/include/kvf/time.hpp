#pragma once
#include <chrono>

using namespace std::chrono_literals;

namespace kvf {
using Seconds = std::chrono::duration<float>;
using Clock = std::chrono::steady_clock;

struct DeltaTime {
	void reset() { start = Clock::now(); }

	auto tick() -> Seconds {
		auto const now = Clock::now();
		dt = Seconds{now - start};
		start = now;
		return dt;
	}

	Clock::time_point start{Clock::now()};
	Seconds dt{};
};
} // namespace kvf
