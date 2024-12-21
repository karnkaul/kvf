#pragma once
#include <stdexcept>

namespace kvf {
class Error : public std::runtime_error {
	using std::runtime_error::runtime_error;
};
} // namespace kvf
