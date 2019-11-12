//
//  neuralNetwork.h
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#pragma once


#include "matrix_gpu.h"

// neural network hidden layer class
class NeuralNetwork {
public:
	NeuralNetwork(const std::vector<int>& neurons, float rate = 0.9f  )
		: H{}, W{}, wT{}, hT{}, HWB{}, actPrimeHWB{}, B{}, dEdB{}, dEdW{} 
	{
		assert(neurons.size() >= 3);

		const auto numParams = neurons.size();
		const auto outputs = neurons[numParams - 1];
		hiddenLayers = numParams - 2;
		totalLayers = numParams - 1;
		learningRate = rate;

		for (auto i = 0; i < totalLayers; ++i) {
			const auto inputs = neurons[i];
			const auto hidden = neurons[static_cast<size_t>(i) + 1];
			

			if (i == 0) {
				H.push_back(Matrix(1, inputs));
				hT.push_back(Matrix(inputs, 1));
				stepY.push_back(Matrix(1, outputs));
			}
			H.push_back(Matrix(1, hidden));
			hT.push_back(Matrix(hidden, 1));
			W.push_back(Matrix(inputs, hidden));
			W[i].randomize();
			wT.push_back(Matrix(hidden, inputs));
			B.push_back(Matrix(1, hidden));
			B[i].randomize();
			HWB.push_back(Matrix(1, hidden));
			actPrimeHWB.push_back(Matrix(1, hidden));
			dEdW.push_back(Matrix(inputs, hidden));
			dEdB.push_back(Matrix(1, hidden));
		}
	}
	
	// propagates forward with the matrix X (1,inputs) - current training input 
	void propagateForward(const Matrix& X) {

		H[0] = X;

		if (DEBUG) {
			std::cout << "Propagate forward:\n";
			H[0].print("H", 0);
		}

		for (int64_t i = 0; i < totalLayers; ++i) {
			HWB[i].computeDotProductOf(H[i], W[i]).add(B[i]);
			H[i + 1L].computeSigmoidOf(HWB[i]);

			if (DEBUG) {
				std::cout << "Layer " << i << ":\n";
				HWB[i].print("HWB", i);
				H[i + 1L].print("H", i + 1L);
				std::cout << std::endl;
			}
		}
	}

	void getCurrentOutput(std::vector<float>& output) {
		stepY[0] = H[totalLayers];
		stepY[0].step();
		stepY[0].toVector(output);
	}

	// propagates backward with the matrix Y_ (1,outputs) - current dataset output
	void propagateBackward(const Matrix& Y_) {
		actPrimeHWB[hiddenLayers].computeSigmoidPrimeOf(HWB[hiddenLayers]);
		dEdB[hiddenLayers] = H[totalLayers];
		dEdB[hiddenLayers].sub(Y_).mul(actPrimeHWB[hiddenLayers]);

		if (DEBUG) {
			std::cout << "Propagate backward:\n";
			actPrimeHWB[hiddenLayers].print("prime(HWB)", hiddenLayers);
			dEdB[hiddenLayers].print("dEdB", hiddenLayers);
			std::cout << std::endl;
		}

		for (int64_t i = static_cast<int64_t>(hiddenLayers) - 1; i >= 0; --i) {
			wT[i + 1].computeTransposeOf(W[i + 1]);
			dEdB[i].computeDotProductOf(dEdB[i + 1], wT[i + 1]);
			actPrimeHWB[i].computeSigmoidPrimeOf(HWB[i]);
			dEdB[i].mul(actPrimeHWB[i]);

			if (DEBUG) {
				std::cout << "Compute dE/dB, pass: " << i << ":\n";
				dEdB[i + 1].print("dEdB", i + 1);
				wT[i + 1].print("wT", i + 1);
				dEdB[i].print("dEdB", i);
			}
		}

		for (int64_t i = 0; i < totalLayers; ++i)
		{
			hT[i].computeTransposeOf(H[i]);
			dEdW[i].computeDotProductOf(hT[i],dEdB[i]);

			if (DEBUG) {
				std::cout << "Compute dE/dW, pass: " << i << ":\n";
				hT[i].print("hT", i);
				dEdW[i].print("dEdW", i);
			}
		}

		for (auto i = 0; i < totalLayers; ++i) {
			W[i].sub(dEdW[i], learningRate);
			B[i].sub(dEdB[i], learningRate);

			if (DEBUG) {
				std::cout << "Substract deltas, pass: " << i << ":\n";
				W[i].print("W", i);
				B[i].print("B", i);
			}
		}
	}

	// export to a RAM vector
	std::vector<std::vector<float>> toVector() const {
		std::vector<std::vector<float>> output = {};
		for (const auto& mat : H)
			output.push_back(mat.toVector());
		for (const auto& mat : W)
			output.push_back(mat.toVector());
		for (const auto& mat : B)
			output.push_back(mat.toVector());
		for (const auto& mat : dEdW)
			output.push_back(mat.toVector());
		for (const auto& mat : dEdB)
			output.push_back(mat.toVector());
		return output;
	}

	// returns the network size in bytes
	size_t getSizeInMemory() {
		size_t result = 0;
		for (const auto& mat : H)
			result += mat.memSize();
		for (const auto& mat : W)
			result += mat.memSize();
		for (const auto& mat : B)
			result += mat.memSize();
		for (const auto& mat : dEdW)
			result += mat.memSize();
		for (const auto& mat : dEdB)
			result += mat.memSize();
		return result;
	}

	// save to a file
	void saveToFile(const std::string filename) {
		std::ofstream fout (filename, std::ios::out);
		
		if (fout) {
			for (const auto& mat : H)
				mat.write(fout);
			for (const auto& mat : W)
				mat.write(fout);
			for (const auto& mat : B)
				mat.write(fout);
			for (const auto& mat : dEdW)
				mat.write(fout);
			for (const auto& mat : dEdB)
				mat.write(fout);
			
			fout.close();
		}
		else {
			std::cout << "failed to open " << filename << " for writing!" << std::endl;
		}
	}


private:
	bool DEBUG = false;

	int hiddenLayers;
	int totalLayers;
	float learningRate;

	// saveable data
	std::vector<Matrix> H;					// (1, hidden) hidden
	std::vector<Matrix> W;					// (inputs, hidden) weights
	std::vector<Matrix> B;					// (1, hidden) bias
	std::vector<Matrix> dEdW;				// (inputs, hidden) dError / dW
	std::vector<Matrix> dEdB;				// (1, hidden) dError / dB

	// temp data
	std::vector<Matrix> wT;					// (hidden, inputs) transpose of W
	std::vector<Matrix> hT;					// (hidden, 1) transpose of H
	std::vector<Matrix> HWB;				// (1, hidden) H dot W + B
	std::vector<Matrix> actPrimeHWB;		// (1, hidden) activation_prime(H dot W + B)
	std::vector<Matrix> stepY;				// (1, outputs) a temp matrix for a current step-refined output
};



