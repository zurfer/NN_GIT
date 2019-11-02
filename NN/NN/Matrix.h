//
//  Matrix.h
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

#include "MyMath.h"


template <typename T>
class Matrix {
public:
    // default ctor
	Matrix () = default;
    
	// ctor setting the dimensions
    Matrix (int r, int c) {
		rows = static_cast<size_t>(r);
		cols = static_cast<size_t>(c);
		vec.resize(rows * cols);
		std::fill(vec.begin(), vec.end(), static_cast<T>(0));
    }
    
	// copy ctor
	Matrix (const Matrix& rhs) {
        vec = rhs.vec;
        rows = rhs.rows;
        cols = rhs.cols;
    }

	// ctor from a vector
	Matrix(int r, int c, std::vector<T> src) {
		rows = static_cast<size_t>(r);
		cols = static_cast<size_t>(c);
		std::copy(src.begin(), src.end(), std::back_inserter(vec));
	}

	// assignment op
    Matrix& operator = (const Matrix& rhs) {
        if (this != &rhs) {
            vec = rhs.vec;
            rows = rhs.rows;
            cols = rhs.cols;
        }
		return *this;
    }
	
	// the number of rows getter
	size_t num_rows() const { return rows; }
	// the number of columns getter
	size_t num_cols() const { return cols; }
	// index op getter
	T operator[] (size_t i) const { return (vec[i]); }

	// convert from a matrix with a different numeric type 
	template <class U>
	Matrix& convert_from(const Matrix<U>& rhs) {
		rows = rhs.num_rows();
		cols = rhs.num_cols();
		size_t num_elements = rows * cols;
		vec.resize(num_elements);
		for (size_t i = 0; i < num_elements; ++i)
			vec[i] = static_cast<T>(rhs[i]);
		return *this;
	}

	// randomize the elements of the matrix
	void randomize() {
		std::generate(vec.begin(), vec.end(), rnd );
	}

	// print the elements of the matrix and its dimensions
	void print() {
		std::cout << "Matrix [" << rows << "x" << cols << "]\n";
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				std::cout <<  std::setw(10) << std::setprecision(6) << vec[j + i * cols];
			}
			std::cout << "\n";
		}
		for(size_t i = 0; i < cols * 10; ++i)
			std::cout << "-";
		std::cout << std::endl;
	}

private:
	struct randomGen {
		T operator () () { return random01<T>(); }
	} rnd;

    std::vector<T> vec {};
    size_t rows {0};
    size_t cols {0};
};
