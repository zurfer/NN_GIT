//
//  utils.h
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <boost/compute.hpp>

namespace compute = boost::compute;

template <typename T>
void printVector(const std::vector<std::vector<T>>& vec, const std::string header = "Matrix", const size_t maxEntries = 100) {

	size_t rows = vec.size();
	size_t cols = (vec.size() > 0 ? vec[0].size() : 0);

	std::cout << header << " [" << rows << "x" << cols << "]\n";

	size_t counter = 0;
	for (const auto& row : vec) {
		for (const auto& col : row) {
			++counter;
			if (counter <= maxEntries) {
				std::cout << std::fixed << std::setw(10) << std::setprecision(2) << std::setfill(' ') << col;
			}
			else {
				std::cout << "<skipped>\n";
				return;
			}
		}
		std::cout << "\n";
	}
	for (size_t i = 0; i < cols * 10; ++i)
		std::cout << "-";
	std::cout << std::endl;
}

template <typename T>
std::string makeReadable(T bytes) {
	static const std::vector<std::string> endings = { "B", "KB", "MB", "GB", "TB" };
	T i;
	for (i = 0; i < endings.size() && bytes >= static_cast<T>(1024); ++i, bytes /= static_cast<T>(1024));
	return std::to_string(bytes) + endings[i];
}


void printPercentage(const size_t n, const size_t max, const bool reset = false, const std::string ending = "%") {
	static size_t currentPercent = 0;
	static bool firstAfterReset = false;

	auto percent = (n + 1) * 100 / max;

	if (reset) {
		currentPercent = 0;
		firstAfterReset = true;
		return;
	}

	if (currentPercent == percent)
		return;

	if (firstAfterReset) {
		firstAfterReset = false;
	}
	else {
		if (currentPercent < 10)
			std::cout << "\b\b";
		else if (currentPercent < 100)
			std::cout << "\b\b\b";
		else
			std::cout << "\b\b\b\b";
	}

	std::cout << percent << ending;
	currentPercent = percent;
	std::cout.flush();
}

