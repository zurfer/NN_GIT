//
//  kernels.h
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#pragma once

#include <boost/compute.hpp>

#include "gpu.h"

namespace compute = boost::compute;
using compute::uint_;
using compute::float_;

namespace OPENCL {

	constexpr auto BLOCK_DIM = 16;

	enum class Program
	{
		Transpose,
		TransposeNaive,
		Sigmoid,
		ForwardSigmoid,
		SigmoidPrime,
		ForwardSigmoidPrime,
		StepFunction,
		ForwardStepFunction,
		MatMulVec,
		DotProduct,
		AddVector,
		SubVector,
		HadamardProduct
	};

	typedef struct {
		const char* source;
		const char* name;
		bool exportBlockDim;
	} ProgramSource;

	using ProgramSourcesMap = std::map<Program, ProgramSource>;
	using CompiledProgramsMap = std::map<Program, compute::kernel>;

	//------------------------------------------------------------------------------------------------

	const char transposeNaive[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void transposeNaive(__global const float* src, __global float* dst, uint width, uint height)
	{
		uint x_index = get_global_id(0);
		uint y_index = get_global_id(1);
		if (x_index < width && y_index < height)
		{
			uint index_in = x_index + width * y_index;
			uint index_out = y_index + height * x_index;
			dst[index_out] = src[index_in];
		}
	}
	);

	const char transposeBlk[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void transposeBlk(__global float* idata, __global float* odata, unsigned int width, unsigned int height, __local float* block)
	{

		// read the matrix tile into shared memory
		unsigned int xIndex = get_global_id(0);
		unsigned int yIndex = get_global_id(1);

		if ((xIndex < width) && (yIndex < height))
		{
			unsigned int index_in = yIndex * width + xIndex;
			block[get_local_id(1) * (BLOCK_DIM + 1) + get_local_id(0)] = idata[index_in];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// write the transposed matrix tile to global memory
		xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
		yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
		if ((xIndex < height) && (yIndex < width))
		{
			unsigned int index_out = yIndex * height + xIndex;
			odata[index_out] = block[get_local_id(0) * (BLOCK_DIM + 1) + get_local_id(1)];
		}
	}
	);

	const char sigmoid[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void sigmoid(const uint N, __global float* inout)
	{
		const uint index = get_global_id(0);
		if (index >= N)
			return;
		inout[index] = 1.0 / (1.0 + exp(-inout[index]));
	}
	);

	const char forwardSigmoid[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void forwardSigmoid(const uint N, __global const float* src, __global float* dst)
	{
		const uint index = get_global_id(0);
		if (index >= N)
			return;
		dst[index] = 1.0 / (1.0 + exp(-src[index]));
	}
	);

	const char sigmoidPrime[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void sigmoidPrime(const uint N, __global float* inout)
	{
		const uint index = get_global_id(0);
		if (index >= N)
			return;
		const float exp_m_x = exp(-inout[index]);
		inout[index] = exp_m_x / (pow(1.0 + exp_m_x, 2.0));
	}
	);

	const char forwardSigmoidPrime[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void forwardSigmoidPrime(const uint N, __global const float* src, __global float* dst)
	{
		const uint index = get_global_id(0);
		if (index >= N)
			return;
		const float exp_m_x = exp(-src[index]);
		dst[index] = exp_m_x / (pow(1.0 + exp_m_x, 2.0));
	}
	);

	const char stepFunction[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void stepFunction(const uint N, __global float* inout)
	{
		const uint index = get_global_id(0);
		if (index >= N)
			return;
		const float x = inout[index];
		inout[index] = x < 0.1 ? 0 : (x > 0.9 ? 1 : x);
	}
	);

	const char forwardStepFunction[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void forwardStepFunction(const uint N, __global const float* src, __global float* dst)
	{
		const uint index = get_global_id(0);
		if (index >= N)
			return;
		const float x = src[index];
		dst[index] = x < 0.1 ? 0 : (x > 0.9 ? 1 : x);
	}
	);


	// OpenCL Kernel Function for element by element vector addition
	const char addVector[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void addVector(__global float* a, __global const float* b, float bScale, uint iNumElements)
	{
		// get index into global data array
		const uint iGID = get_global_id(0);

		// bound check 
		if (iGID >= iNumElements) 
			return;

		// add the vector elements
		a[iGID] += (b[iGID] * bScale);
	}
	);

	// OpenCL Kernel Function for element by element vector addition
	const char subVector[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void subVector(__global float* a, __global const float* b, float bScale, uint iNumElements)
	{
		// get index into global data array
		const uint iGID = get_global_id(0);

		// bound check 
		if (iGID >= iNumElements)
			return;

		// add the vector elements
		a[iGID] -= (b[iGID] * bScale);
	}
	);

	// OpenCL Kernel Function for element by element vector addition
	const char hadamardProduct[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void hadamardProduct(__global float* a, __global const float* b, float bScale, uint iNumElements)
	{
		// get index into global data array
		const uint iGID = get_global_id(0);

		// bound check 
		if (iGID >= iNumElements)
			return;

		// add the vector elements
		a[iGID] *= (b[iGID] * bScale);
	}
	);


	const char matrixMulVector[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void matrixMulVector(__global const float* M,
			 __global const float* V, __global float* W,
			uint width, uint height,
			__local float* partialDotProduct)
	{
		for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {
			// Row pointer
			const __global float* row = M + y * width;

			// Each work-item accumulates as many products as necessary
			// into local variable "sum"
			float sum = 0;
			for (uint x = get_local_id(0); x < width; x += get_local_size(0))
				sum += row[x] * V[x];

			// Each partial dot product is stored in shared memory
			partialDotProduct[get_local_id(0)] = sum;
			barrier(CLK_LOCAL_MEM_FENCE);

			if (get_local_id(0) == 0) {
				float dotProduct = 0;
				for (uint t = 0; t < get_local_size(0); ++t)
					dotProduct += partialDotProduct[t];
				W[y] = dotProduct;
			}

			// Synchronize to make sure the first work-item is done with
			// reading partialDotProduct
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	);

	const char matMul0[] =
		"__kernel void dotProd(__global float* A, __global float* B, __global float* C,		\n"
		"	__local float* As, __local float* Bs, int uiWA, int uiWB, int trueLocalSize1)	\n"
		"{	\n"
		"int bx = get_group_id(0);				\n"
		"int by = get_group_id(1);				\n"
		"int tx = get_local_id(0);				\n"
		"int ty = get_local_id(1);				\n"
		"int aBegin = uiWA * BLOCK_DIM * by;	\n"
		"int aEnd = aBegin + uiWA - 1;			\n"
		"int aStep = BLOCK_DIM;					\n"
		"int bBegin = BLOCK_DIM * bx;			\n"
		"int bStep = BLOCK_DIM * uiWB;			\n"
		"float Csub = 0.0f;						\n"
		"for (int a = aBegin, b = bBegin;		\n"
		"	a <= aEnd;							\n"
		"	a += aStep, b += bStep) {			\n"
		"	As[tx + ty * BLOCK_DIM] = A[a + uiWA * ty + tx];	\n"	
		"	Bs[tx + ty * BLOCK_DIM] = B[b + uiWB * ty + tx];	\n" 
		"	barrier(CLK_LOCAL_MEM_FENCE);		\n"
		"	#pragma unroll						\n"
		"	for (int k = 0; k < BLOCK_DIM; ++k)	\n"
		"		Csub += As[k + ty * BLOCK_DIM] * Bs[tx + k * BLOCK_DIM];	\n"
		"	barrier(CLK_LOCAL_MEM_FENCE);									\n"
		"}	\n"
		"if (get_global_id(1) < trueLocalSize1)	\n"
		"	C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;			\n"
		"}	\n";

	// Matrices in column-major format
	// A: M rows x K columns,
	// B: K rows x N columns,
	// C: M rows x N columns, 
	//                         
	//                   N     
	//                o-----o  
	//                |     |  
	//              K | [B] |  
	//                |     |  
	//                o-----o  
	//        K          N     
	//    o-------o   o-----o  
	//  M |  [A]  | M | [C] |  
	//    |       |   |     |  
	//    o-------o   o-----o  
	const char matrixMulMatrix[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void matrixMulMatrix(const uint M, const uint N, const uint K,
			__global const float* A, __global const float* B, __global float* C)
	{
		// Thread identifiers
		const int globalRow = get_global_id(0); // Row ID of C (0..M)
		const int globalCol = get_global_id(1); // Col ID of C (0..N)

		
		if (globalRow < M && globalCol < N) {
			// Compute a single element (loop over K)
			float acc = 0.0f;
			for (int k = 0; k < K; k++) {
				//acc += A[k * M + globalRow] * B[globalCol * K + k];
				acc += A[M * globalRow + k] * B[N * k + globalCol];
				
			}
			C[globalRow * N + globalCol] = acc;
		}
	}
	);

	const char matrixMulMatrixNaive[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void matrixMulMatrixNaive(const uint width, const uint height, const uint length,
			__global const float* A, __global const float* B, __global float* C)
	{
		// Thread identifiers
		const int x_index = get_global_id(0);	// (0..width)
		const int y_index = get_global_id(1);	// (0..height)
		// check the boundaries
		if (y_index < height && x_index < width) {
			// Compute a single element (loop over length)
			float acc = 0.0f;
			for (int k = 0; k < length; k++) {
				acc += A[k + y_index * length] * B[x_index + k * width];

			}
			C[y_index * width + x_index] = acc;
		}
	}
	);

	//------------------------------------------------------------------------------------------------

	ProgramSourcesMap sourcesPool = {
		{ Program::Transpose,			{ transposeBlk,			"transposeBlk",			true	}},
		{ Program::TransposeNaive,		{ transposeNaive,		"transposeNaive",		false	}},
		{ Program::Sigmoid,				{ sigmoid,				"sigmoid",				false	}},
		{ Program::SigmoidPrime,		{ sigmoidPrime,			"sigmoidPrime",			false	}},
		{ Program::StepFunction,		{ stepFunction,			"stepFunction",			false	}},
		{ Program::ForwardSigmoid,		{ forwardSigmoid,		"forwardSigmoid",		false	}},
		{ Program::ForwardSigmoidPrime,	{ forwardSigmoidPrime,	"forwardSigmoidPrime",	false	}},
		{ Program::ForwardStepFunction,	{ forwardStepFunction,	"forwardStepFunction",	false	}},
		{ Program::MatMulVec,			{ matrixMulVector,		"matrixMulVector",		false	}},
		{ Program::DotProduct,			{ matrixMulMatrixNaive,	"matrixMulMatrixNaive",	true	}},
		{ Program::AddVector,			{ addVector,			"addVector",			false	}},
		{ Program::SubVector,			{ subVector,			"subVector",			false	}},
		{ Program::HadamardProduct,		{ hadamardProduct,		"hadamardProduct",		false	}}
	};

	CompiledProgramsMap kernelsPool;

	//------------------------------------------------------------------------------------------------
	
	// Compile all programs in the sources pool and save the results to the kernels pool
	void compilePrograms(compute::context& context) {
		for (const auto& p : sourcesPool) {
			compute::program program;
			if (p.second.exportBlockDim) {
				std::stringstream options;
				options << "-DBLOCK_DIM=" << BLOCK_DIM;		// setup compilation flags
				try {
					program = compute::program::build_with_source(p.second.source, context, options.str());
				}
				catch (...) {
					std::cout << "Compiling the <" << p.second.name << "> kernel with the options " << options.str() << " has failed!" << std::endl;
					continue;
				}
			}
			else {
				try {
					program = compute::program::build_with_source(p.second.source, context);
				}
				catch (...) {
					std::cout << "Compiling the <" << p.second.name << "> kernel has failed!" << std::endl;
					continue;
				}
			}
			
			std::pair <CompiledProgramsMap::iterator, bool> ptr;
			ptr = kernelsPool.emplace(p.first, program.create_kernel(p.second.name));
			if (!ptr.second) {
				std::cout << "Failed to insert the compiled program into the map: the key already exists." << std::endl;
			}
		}
	}

	// round up just above the multiply of localWorkSize
	size_t shrRoundUp(size_t localWorkSize, size_t numItems) {
		size_t result = localWorkSize;
		while (result < numItems)
			result += localWorkSize;

		return result;
	}

	//  a = a (vector op) b
	void applyTransform(Program prog, compute::vector<float>& a, const compute::vector<float>& b, const float bScale = 1.0f) {
		auto& gpu = GPU::defGPU();
		
		auto numElements = std::min(a.size(), b.size());
		size_t szLocalWorkSize = 256;
		size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, numElements);
		
		kernelsPool[prog].set_arg(0, a.get_buffer());
		kernelsPool[prog].set_arg(1, b.get_buffer());
		kernelsPool[prog].set_arg(2, static_cast<float_>(bScale));
		kernelsPool[prog].set_arg(3, static_cast<uint_>(numElements));
		
		auto dummy = gpu.queue.enqueue_nd_range_kernel(kernelsPool[prog], 1, NULL, &szGlobalWorkSize, &szLocalWorkSize);
		gpu.queue.finish();
	}

	// out = program (in)
	void forwardProgram(Program prog, compute::vector<float>& in, compute::vector<float>& out) {
		auto& gpu = GPU::defGPU();
		size_t globalWorkSize = std::min(in.size(), out.size());

		kernelsPool[prog].set_arg(0, static_cast<uint_>(globalWorkSize));
		kernelsPool[prog].set_arg(1, in.get_buffer());
		kernelsPool[prog].set_arg(2, out.get_buffer());
		auto dummy = gpu.queue.enqueue_1d_range_kernel(kernelsPool[prog], 0, globalWorkSize, 0);
		gpu.queue.finish();
	}

	// in = program (in)
	void applyProgram(Program prog, compute::vector<float>& inout) {
		auto& gpu = GPU::defGPU();
		size_t globalWorkSize = inout.size();

		kernelsPool[prog].set_arg(0, static_cast<uint_>(globalWorkSize));
		kernelsPool[prog].set_arg(1, inout.get_buffer());
		auto dummy = gpu.queue.enqueue_1d_range_kernel(kernelsPool[prog], 0, globalWorkSize, 0);
		gpu.queue.finish();
	}

	// out = transpose (in)
	void transpose(compute::vector<float>& in, compute::vector<float>& out, unsigned int sizeX, unsigned int sizeY) {
		auto& gpu = GPU::defGPU();

		const size_t localWorkSize[2] = { BLOCK_DIM, BLOCK_DIM };
		const size_t globalWorkSize[2] = { shrRoundUp(BLOCK_DIM, sizeX), shrRoundUp(BLOCK_DIM, sizeY) };

		kernelsPool[Program::Transpose].set_arg(0, in.get_buffer());
		kernelsPool[Program::Transpose].set_arg(1, out.get_buffer());
		kernelsPool[Program::Transpose].set_arg(2, static_cast<uint_>(sizeX));
		kernelsPool[Program::Transpose].set_arg(3, static_cast<uint_>(sizeY));
		kernelsPool[Program::Transpose].set_arg(4, (BLOCK_DIM + 1) * BLOCK_DIM * sizeof(float), 0);
		auto dummy = gpu.queue.enqueue_nd_range_kernel(kernelsPool[Program::Transpose], 2, NULL, globalWorkSize, localWorkSize);
		gpu.queue.finish();
	}

	// out = inMatrix dotprod inVec
	void matMulVec(compute::vector<float>& inmat, compute::vector<float>& invec, compute::vector<float>& out, unsigned int sizeX, unsigned int sizeY) {
		auto& gpu = GPU::defGPU();
		
		size_t szLocalWorkSize = 16;
		size_t szGlobalWorkSize = static_cast<size_t>(gpu.num_compute_units) * 2UL * szLocalWorkSize;

		kernelsPool[Program::MatMulVec].set_arg(0, inmat.get_buffer());
		kernelsPool[Program::MatMulVec].set_arg(1, invec.get_buffer());
		kernelsPool[Program::MatMulVec].set_arg(2, out.get_buffer());
		kernelsPool[Program::MatMulVec].set_arg(3, static_cast<uint_>(sizeX));
		kernelsPool[Program::MatMulVec].set_arg(4, static_cast<uint_>(sizeY));
		kernelsPool[Program::MatMulVec].set_arg(5, szLocalWorkSize * sizeof(float), 0);
		auto dummy = gpu.queue.enqueue_nd_range_kernel(kernelsPool[Program::MatMulVec], 1, NULL, &szGlobalWorkSize, &szLocalWorkSize);
		gpu.queue.finish();
	}


	// Matrices in column-major format
	// A: width x length
	// B: lenght x height
	// C: width x height 
	// c = a dotprod b
	void dotProduct(const compute::vector<float>& a, const compute::vector<float>& b, compute::vector<float>& c, 
		unsigned int width, unsigned int height, unsigned int length) {
		auto& gpu = GPU::defGPU();

		size_t localWorkSize[] =  { BLOCK_DIM, BLOCK_DIM };
		size_t globalWorkSize[] = { shrRoundUp(BLOCK_DIM, width), shrRoundUp(BLOCK_DIM, height) };

		kernelsPool[Program::DotProduct].set_arg(0, static_cast<uint_>(width));
		kernelsPool[Program::DotProduct].set_arg(1, static_cast<uint_>(height));
		kernelsPool[Program::DotProduct].set_arg(2, static_cast<uint_>(length));
		kernelsPool[Program::DotProduct].set_arg(3, a.get_buffer());
		kernelsPool[Program::DotProduct].set_arg(4, b.get_buffer());
		kernelsPool[Program::DotProduct].set_arg(5, c.get_buffer());
		//kernelsPool[Program::DotProduct].set_arg(6, BLOCK_DIM * BLOCK_DIM * sizeof(float), 0);
		//kernelsPool[Program::DotProduct].set_arg(7, BLOCK_DIM * BLOCK_DIM * sizeof(float), 0);
		auto dummy = gpu.queue.enqueue_nd_range_kernel(kernelsPool[Program::DotProduct], 2, NULL, globalWorkSize, localWorkSize);
		gpu.queue.finish();
	}


} // end of the OPENCL namespace