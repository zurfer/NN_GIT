//
//  matrix_gpu.h
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <boost/compute.hpp>
#include "gpu.h"
#include "kernels.h"

namespace compute = boost::compute;
using compute::uint_;

class Matrix {
public:

	// Default ctor
	Matrix() = delete;
	//Matrix(GPU::ComputeKernel& k = GPU::defaultGPU()) : rows{ 0 }, cols{ 0 }, vec(0, k.context), kernel{ k }, printDebug{ false }  {
	//}

	// ctor: allocates a gpu vector
	Matrix(size_t r, size_t c, GPU::Device& dev = GPU::defGPU()) 
		: rows{ r }, cols{ c }, vec(r * c, dev.context), device{ dev }
	{
		device.queue.finish();
	}

	// copy ctor
	Matrix(const Matrix& rhs, GPU::Device& dev = GPU::defGPU()) 
		: rows{ rhs.rows }, cols{ rhs.cols }, vec(rhs.vec), device{ dev }
	{
		device.queue.finish();
	}

	// move ctor
	Matrix(Matrix&& rhs) noexcept 
		: rows{ rhs.rows }, cols{ rhs.cols }, vec(std::move(rhs.vec)), device{ rhs.device }
	{
	}

	// ctor: copy from a vector
	Matrix(const std::vector<float>& rhs, GPU::Device& k = GPU::defGPU()) 
		: rows{ 1 }, cols{ rhs.size() }, vec(rhs.size(), k.context), device{ k }
	{
		auto dummy = compute::copy(rhs.cbegin(), rhs.cend(), vec.begin(), device.queue);
		device.queue.finish();
	}

	// returns the number of rows
	unsigned int height() const { 
		return static_cast<unsigned int>(rows); 
	}

	// returns the number of columns
	unsigned int width() const { 
		return static_cast<unsigned int>(cols);
	}

	// return the amount of GPU memory (in bytes) occupied by vec
	size_t memSize() const {
		return vec.size() * sizeof(float);
	}

	// copy from a matrix
	Matrix& operator = (const Matrix& rhs) {
		assert(rows == rhs.rows && cols == rhs.cols);
		auto dummy = compute::copy(rhs.vec.cbegin(), rhs.vec.cend(), vec.begin(), device.queue);
		device.queue.finish();
		return *this;
	}

	// copy from a vector
	Matrix& operator = (const std::vector<float>& rhs) {
		assert(rhs.size() == vec.size());
		auto dummy = compute::copy(rhs.cbegin(), rhs.cend(), vec.begin(), device.queue);
		device.queue.finish();
		return *this;
	}

	// copy from a RAM vector of vectors
	Matrix& operator = (const std::vector<std::vector<float>>& rhs) {
		auto r = rhs.size();
		auto c = (r > 0) ? rhs[0].size() : 0;
		assert(rows == r && cols == c);
		
		size_t offset = 0;
		for (const auto& row : rhs) {
			auto dummy = compute::copy(row.cbegin(), row.cend(), vec.begin() + offset, device.queue);
			device.queue.finish();
			offset += c;
		}
		return *this;
	}
	
	// add scalar
	Matrix& add (float scalar) {
		using compute::lambda::_1;
		auto dummy = compute::transform(vec.begin(), vec.end(), vec.begin(), _1 + scalar, device.queue);
		device.queue.finish();
		return *this;
	}

	// sub scalar
	Matrix& sub (float scalar) {
		using compute::lambda::_1;
		auto dummy = compute::transform(vec.begin(), vec.end(), vec.begin(), _1 - scalar, device.queue);
		device.queue.finish();
		return *this;
	}

	// mul by scalar
	Matrix& mul (float scalar) {
		using compute::lambda::_1;
		auto dummy = compute::transform(vec.begin(), vec.end(), vec.begin(), _1 * scalar, device.queue);
		device.queue.finish();
		return *this;
	}

	// add matrix
	Matrix& add (const Matrix& rhs, const float scale = 1.0f) {
		assert(rhs.vec.size() == vec.size());
		OPENCL::applyTransform(OPENCL::Program::AddVector, vec, rhs.vec, scale);
		return *this;
	}

	// sub matrix
	Matrix& sub (const Matrix& rhs, const float scale = 1.0f) {
		assert(rhs.vec.size() == vec.size());
		OPENCL::applyTransform(OPENCL::Program::SubVector, vec, rhs.vec, scale);
		return *this;
	}

	// mul by matrix (elements by elements)
	Matrix& mul (const Matrix& rhs, const float scale = 1.0f) {
		assert(rhs.vec.size() == vec.size());
		OPENCL::applyTransform(OPENCL::Program::HadamardProduct, vec, rhs.vec, scale);
		return *this;
	}

	// apply the random 0..1 to the elements
	Matrix& randomize() {
		device.rnd_dist.generate(vec.begin(), vec.end(), device.random_engine, device.queue);
		device.queue.finish();
		return *this;
	}

	// fill the elements with a value
	Matrix& fill (float value = 0.f ) {
		compute::fill(vec.begin(), vec.end(), value, device.queue);
		device.queue.finish();
		return *this;
	}

	// fill the elements with a sequence
	Matrix& iota(float start = 0.f) {
		compute::iota(vec.begin(), vec.end(), start, device.queue);
		device.queue.finish();
		return *this;
	}

	Matrix& step() {
		OPENCL::applyProgram(OPENCL::Program::StepFunction, vec);
		return *this;
	}

	// calc and store sigmoid
	Matrix& computeSigmoidOf(Matrix& rhs) {
		assert(rhs.vec.size() == vec.size());
		OPENCL::forwardProgram(OPENCL::Program::ForwardSigmoid, rhs.vec, vec);
		return *this;
	}

	// calc and store sigmoid prime
	Matrix& computeSigmoidPrimeOf(Matrix& rhs) {
		assert(rhs.vec.size() == vec.size());
		OPENCL::forwardProgram(OPENCL::Program::ForwardSigmoidPrime, rhs.vec, vec);
		return *this;
	}

	// compute dot product of a and b and store the results
	Matrix& computeDotProductOf(Matrix& a, Matrix& b) {
		if (a.cols != b.rows) {
			std::cout << "\nDotProduct: the number of columns in the first argument (" << a.cols << ") does not match the number of rows in the second (" << b.rows << ")." << std::endl;
			return *this;
		}
		if (rows != a.rows || cols != b.cols) {
			std::cout << "\nDotProduct: This matrix must have the [" << a.rows << "x" << b.cols << "] dimensions to be able to store the result of the dot product." << std::endl;
			return *this;
		}
		OPENCL::dotProduct(a.vec, b.vec, vec, width(), height(), static_cast<unsigned int>(b.rows));
		return *this;
	}

	// export the matrix to a RAM vector
	std::vector<float> toVector() const {
		std::vector<float> result (vec.size());
		auto dummy = compute::copy(vec.begin(), vec.end(), result.begin(), device.queue);
		device.queue.finish();
		return result;
	}

	// export the matrix to a RAM vector
	void toVector(std::vector<float>& result) const {
		assert(result.size() == vec.size());
		//result.resize(vec.size());
		auto dummy = compute::copy(vec.begin(), vec.end(), result.begin(), device.queue);
		device.queue.finish();
	}

	// multitply a matrix by a row matrix and store the result
	Matrix& computeMatrixMulVector(Matrix& matrix, Matrix& vector) {
		assert(matrix.height() == vec.size() && matrix.width() == vector.vec.size());
		OPENCL::matMulVec(matrix.vec, vector.vec, vec, matrix.width(), matrix.height());
		return *this;
	}

	// multitply a matrix by a vector and store the result
	Matrix& computeMatrixMulVector(Matrix& matrix, compute::vector<float>& vector) {
		assert(matrix.height() == vec.size() && matrix.width() == vector.size());
		OPENCL::matMulVec(matrix.vec, vector, vec, matrix.width(), matrix.height());
		return *this;
	}

	// transpose and store the results
	Matrix& computeTransposeOf( Matrix& rhs) {
		if (rhs.vec.size() != vec.size()) {
			std::cout << "Transpose: the dimensions of the vectors do not match." << std::endl;
			std::cout << "This: " << rows << "x" << cols << std::endl;
			std::cout << "Argument: " << rhs.rows << "x" << rhs.cols << std::endl;
			return *this;
		}
		
		OPENCL::transpose(rhs.vec, vec, height(), width());
		rows = rhs.cols;
		cols = rhs.rows;
		return *this;
	}
	
	// writes the contents to a file stream
	void write(std::ofstream& out) const {
		out << rows << "\n" << cols  << "\n";
		auto dummy = compute::copy(vec.begin(), vec.end(), std::ostream_iterator<float>(out, "\n"), device.queue);
		device.queue.finish();
		out << "\n";
	}

	// reads a file stream into the contents
	void read(std::ifstream& in) {
		size_t r, c;
		r = in.get();
		c = in.get();
		if (rows != r || cols != c) {
			std::cout << "read: vector dimensions do not match!" << std::endl;
			return;
		}
		size_t numEntries = rows * cols;
		std::vector<float> buf(numEntries);
		for (auto i = 0; i < numEntries; ++i)
			if (in.good())
				in >> buf[i];
		
		auto dummy = compute::copy_n(buf.begin(), numEntries, vec.begin() , device.queue);
		device.queue.finish();
	}



	// prints the vector
	void print(const std::string header = "Matrix", const int64_t index = -1, const size_t maxEntries = 100) {
		auto numEntries = std::min(vec.size(), maxEntries);
		std::vector<float> buf(numEntries);
		auto dummy = compute::copy_n(vec.begin(), numEntries, buf.begin(), device.queue);
		device.queue.finish();

		std::cout << header;
		if (index >= 0)
			std::cout << "[" << index << "]";
		std::cout << ", dimensions: [" << rows << "x" << cols << "]\n";

		size_t counter = 0;
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				++counter;
				if (counter <= numEntries) {
					std::cout << std::fixed << std::setw(10) << std::setprecision(2) << std::setfill(' ') << vec[j + i * cols];
				}
				else {
					std::cout << "<skipped>\n";
					return;
				}
			}
			std::cout << "\n";
		}
		std::cout << std::endl;
	}

private:
	compute::vector<float> vec;		// gpu vector
	GPU::Device& device;			// gpu context
	size_t rows;					// number of rows in the matrix
	size_t cols;					// number of columns in the matrix
	
	
	size_t shrRoundUp(size_t localWorkSize, size_t numItems) {
		size_t result = localWorkSize;
		while (result < numItems)
			result += localWorkSize;

		return result;
	}

	/* 
	// returns the smallest multiple of local work size bigger than minimum
	size_t shrRoundUp(size_t quantization, size_t minimum) {
		size_t size = (minimum / quantization) * quantization;
		if (size < minimum) {
			size += quantization;
		}
		return size;
	}
	

	int getPower2Upperbound(int value) {
		int upperbound = 1;
		while (upperbound < value) {
			upperbound <<= 1;
		}
		return upperbound;
	}
	*/
};

