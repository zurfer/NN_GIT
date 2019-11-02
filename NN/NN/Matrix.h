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
	Matrix () {std::cout << "Default ctor called\n";}
    
	// ctor setting the dimensions
    Matrix (size_t r, size_t c) : rows_{r}, cols_{c} {
		//rows_ = static_cast<size_t>(r);
		//cols_ = static_cast<size_t>(c);
		vec_.resize(rows_ * cols_);
		std::fill(vec_.begin(), vec_.end(), static_cast<T>(0));
        std::cout << "Empty matrix ctor called\n";
    }
    
	// copy ctor
	Matrix (const Matrix& rhs) {
        std::copy(rhs.vec.begin(), rhs.vec.end(), std::back_inserter(vec_));
        //vec = rhs.vec;
        rows_ = rhs.rows;
        cols_ = rhs.cols;
        std::cout << "Copy ctor called\n";
    }

	// ctor from a vector
	Matrix(int r, int c, std::vector<T> src) {
		rows_ = static_cast<size_t>(r);
		cols_ = static_cast<size_t>(c);
		std::copy(src.begin(), src.end(), std::back_inserter(vec_));
        std::cout << "Copy from a vector ctor called\n";
	}

	// assignment op
    Matrix& operator = (const Matrix& rhs) {
        if (this != &rhs) {
            std::copy(rhs.vec.begin(), rhs.vec.end(), std::back_inserter(vec_));
            rows_ = rhs.rows;
            cols_ = rhs.cols;
        }
        std::cout << "Assigment op from the same numeric type called\n";
		return *this;
    }
	
	// the number of rows getter
	size_t num_rows() const { return rows_; }
	// the number of columns getter
	size_t num_cols() const { return cols_; }
	// index op getter
	T operator[] (size_t i) const { return (vec_[i]); }

	// convert from a matrix with a different numeric type 
	template <class U>
	Matrix& operator = (const Matrix<U>& rhs) {
		rows_ = rhs.num_rows();
		cols_ = rhs.num_cols();
		size_t num_elements = rows_ * cols_;
		vec_.resize(num_elements);
		for (size_t i = 0; i < num_elements; ++i)
			vec_[i] = static_cast<T>(rhs[i]);
        std::cout << "Assigment op from other numeric type called\n";
		return *this;
	}
    
    

	// randomize the elements of the matrix
	Matrix& randomize() {
        RandomGen rnd;
		std::generate(vec_.begin(), vec_.end(), rnd);
        return *this;
	}

    Matrix& operator + (const Matrix& rhs) {
        
    }
    
	// print the elements of the matrix and its dimensions
	void print() {
		std::cout << "Matrix [" << rows_ << "x" << cols_ << "]\n";
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < cols_; ++j) {
				std::cout << std::fixed << std::setw(11) << std::setprecision(6) << std::setfill(' ') << vec_[j + i * cols_];
			}
			std::cout << "\n";
		}
		for(size_t i = 0; i < cols_ * 10; ++i)
			std::cout << "-";
		std::cout << std::endl;
	}

private:
    std::vector<T> vec_ {};
    size_t rows_ {0};
    size_t cols_ {0};
    typedef struct {
        T operator () () { return random01<T>(); }
    } RandomGen;
};
