#pragma once

#include <random>

template <typename T> 
T random01() {
	static std::default_random_engine e;
	static std::uniform_real_distribution<T> dis(0, 1); 
	return dis(e);
}
