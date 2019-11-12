//
//  matrix_cpu.h
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <initializer_list>
#include <numeric>
#include <algorithm>
#include <cassert>

#include "random.h"


template <typename T>
class Matrix {
public:
	// the number of rows getter
	size_t num_rows() const { return rows; }
	// the number of columns getter
	size_t num_cols() const { return cols; }
	// index op getter
	T operator[] (size_t i) const { return (vec[i]); }

    // default ctor
	Matrix() : rows{ 0 }, cols{ 0 }, vec{ } {
		std::cout << "Default ctor called\n";
	}
	
	// ctor alloc vector and fill it with zeroes
    Matrix (size_t r, size_t c) : rows{ r }, cols{ c }, vec{ } {
		vec.resize(rows * cols);
		fill(0);
        std::cout << "Allocate [" << rows << "x" << cols << "] matrix and fill it with zeros\n";
    }

	// copy ctor from a vector
	Matrix(size_t r, size_t c, const std::vector<T>& src) : rows{ r }, cols{ c }, vec(src) {
		std::cout << "Copy from a vector ctor called\n";
	}

	// move ctor from a vector
	Matrix(size_t r, size_t c, std::vector<T>&& src) noexcept : rows{ r }, cols{ c }, vec(std::move(src)) {
		std::cout << "Move from a vector ctor called\n";
	}

	// copy ctor
	Matrix (const Matrix& rhs) : rows { rhs.rows }, cols { rhs.cols }, vec(rhs.vec) {
        std::cout << "Copy ctor called\n";
    }

	// move ctor
	Matrix(Matrix&& rhs) noexcept : rows{ rhs.rows }, cols{ rhs.cols }, vec(std::move(rhs.vec)) {
		std::cout << "Move ctor called\n";
	}

	// assignment op from the same numeric type
	Matrix& operator = (const Matrix& rhs) {
		if (this != &rhs) {
			vec = rhs.vec;
			rows = rhs.rows;
			cols = rhs.cols;
			//std::copy(rhs.vec.begin(), rhs.vec.end(), std::back_inserter(vec));
		}
		std::cout << "Assigment op from the same numeric type called\n";
		return *this;
	}

	// assignment op from a different numeric type 
	template <class U>
	Matrix& operator = (const Matrix<U>& rhs) {
		rows = rhs.num_rows();
		cols = rhs.num_cols();
		size_t num_elements = rows * cols;
		vec.resize(num_elements);
		for (size_t i = 0; i < num_elements; ++i)
			vec[i] = static_cast<T>(rhs[i]);
		std::cout << "Assigment op from other numeric type called\n";
		return *this;
	}


	// randomize the elements of the matrix
	Matrix& randomize() {
        RandomGen rnd;
		std::generate(vec.begin(), vec.end(), rnd);
        return *this;
	}

	// fill with a value
	Matrix& fill(T value) {
		std::fill(vec.begin(), vec.end(), static_cast<T>(value));
		return *this;
	}

	// transpose matrix
	Matrix& store_transposed() {

	}

	// add matrices
    Matrix operator + (const Matrix& rhs) const {
		Matrix<T> result;
		std::transform(vec.begin(), vec.end(), rhs.vec.begin(), std::back_inserter(result.vec), std::plus<T>());
		result.rows = rows;
		result.cols = cols;
		return result;
    }

	// add matrices
	Matrix operator - (const Matrix& rhs) const {
		Matrix<T> result;
		std::transform(vec.begin(), vec.end(), rhs.vec.begin(), std::back_inserter(result.vec), std::minus<T>());
		result.rows = rows;
		result.cols = cols;
		return result;
	}

	Matrix operator * (T scalar) const {
		Matrix<T> result;
		std::transform(vec.begin(), vec.end(), std::back_inserter(result.vec), [=](T val) -> T { return val * scalar; });
		result.rows = rows;
		result.cols = cols;
		return result;
	}

	Matrix operator * (const Matrix& rhs) const {
		Matrix<T> result;
		assert(cols == rhs.rows);
		return result;
	}

	// print the elements of the matrix and its dimensions
	void print() const {
		std::cout << "Matrix [" << rows << "x" << cols << "]\n";
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				std::cout << std::fixed << std::setw(11) << std::setprecision(6) << std::setfill(' ') << vec[j + i * cols];
			}
			std::cout << "\n";
		}
		for(size_t i = 0; i < cols * 10; ++i)
			std::cout << "-";
		std::cout << std::endl;
	}

private:
	typedef struct {
		T operator () () { return random01<T>(); }
	} RandomGen;

	std::vector<T> vec;
    size_t rows;
    size_t cols;
};
