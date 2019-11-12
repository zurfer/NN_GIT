//
//  main.cpp
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//
#include <fstream>
#include <iostream>

#include "matrix_gpu.h"
#include "mnist/mnist_reader_less.hpp"
#include "neuralNetwork.h"
#include "utils.h"
#include "kernels.h"


template <typename In, typename Out>
void labelToVector(In label, int outputSize, std::vector<Out>& vec) {
	std::fill(vec.begin(), vec.end(), static_cast<Out>(0));
	auto pos = std::min<int>(outputSize - 1, std::max<int>(static_cast<In>(0), label ));
	vec[pos] = static_cast<Out>(1);
}

template <typename In, typename Out>
void imageToVector(const std::vector<In>& in, std::vector<Out>& out) {
	std::transform(
		in.cbegin(),
		in.cend(),
		out.begin(),
		[=](In x) {return static_cast<Out>(x) / static_cast<Out>(255); }
	);
}

template <typename T>
double meanSquaredError(const std::vector<T>& actual, const std::vector<T>& predicted) {
	double sum = 0.0;
	auto len = std::min(actual.size(), predicted.size());
	for (auto i = 0; i < len; ++i) {
		double diff = static_cast<double>(actual[i]) - static_cast<double>(predicted[i]);
		sum += (diff * diff);
	}
	double result = 1.0 / static_cast<double>(len) * sum;
	return result;
}

bool isHit(const std::vector<float>& predicted, int actual) {
	auto posMax = std::max_element(predicted.cbegin(), predicted.cend());
	auto pred = posMax - predicted.cbegin();
	return (actual == pred);
}


int main(int argc, char *argv[]) {

	int maxImagesToProcess = 0;
	if (argc == 2) {
		maxImagesToProcess = std::stoi(argv[1]);
		std::cout << "Processing first " << maxImagesToProcess << " images\n" << std::endl;
	}

	// normalized images as matrices uploaded to a GPU
	std::vector<Matrix> images = {};
	std::vector<Matrix> testImages = {};
	
	// normalized labels as matrices uploaded to a GPU
	std::vector<Matrix> labels = {};
	std::vector<Matrix> testLabels = {};

	// print GPU info
	GPU::defGPU().print_info();

	// compile opencl programs
	std::cout << "Compiling OpenCL subroutines...  ";
	OPENCL::compilePrograms(GPU::defGPU().context);
	std::cout << "ok\n" << std::endl;

	// Load databases
	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = mnist::read_dataset<uint8_t, uint8_t>();
	if (dataset.training_images.size() == 0 ||
		dataset.training_labels.size() == 0 ||
		dataset.test_images.size() == 0 ||
		dataset.test_labels.size() == 0) {
		std::cout << "Error loading a dataset; exiting." << std::endl;
		return - 1;
	}

	const auto imageSize = dataset.training_images[0].size();		// number of pixels in one image
	const auto imageDim = static_cast<int>(std::sqrt(imageSize));	// image dimensions, width x height
	const auto numTrainingImages = dataset.training_images.size();	// number of training images in the dataset
	const auto numTrainingLabels = dataset.training_labels.size();	// number of training labels in the dataset
	const auto numTestImages = dataset.test_images.size();			// number of training images in the dataset
	const auto numTestLabels = dataset.test_labels.size();			// number of training labels in the dataset
	constexpr auto hiddenSize = 16;									// number of the neurons in a hidden layer
	constexpr auto outputSize = 10;									// number of output categories = number of output neurons

	std::cout << "[MNIST dataset]\n\n";
	std::cout << "The number of Training images:   " << dataset.training_images.size() << "\n";
	std::cout << "The amount of RAM taken:         " << makeReadable(numTrainingImages * imageSize * sizeof(uint8_t)) << "\n";
	std::cout << "Image dimensions:                " << imageDim << "x" << imageDim << "\n\n";
	
	std::cout << "The number of Training labels:   " << dataset.training_labels.size() << "\n";
	std::cout << "The amount of RAM taken:         " << makeReadable(numTrainingLabels * sizeof(uint8_t)) << "\n\n";
	
	std::cout << "The number of Test images:       " << dataset.test_images.size() << "\n";
	std::cout << "The amount of RAM taken:         " << makeReadable(numTestImages * imageSize * sizeof(uint8_t)) << "\n\n";
	
	std::cout << "The number of Test labels:       " << dataset.test_labels.size() << "\n";
	std::cout << "The amount of RAM taken:         " << makeReadable(numTestLabels * sizeof(uint8_t)) << "\n\n" << std::endl;


	// create a temp vector [imageSize],
	// normalize the integer values from the training dataset to float [0; 1] values and fill the temp vector
	// convert the temp vector to a GPU matrix of the size [1, imageSize]
	// store the resulting matrix to <images> vector of matrices

	std::cout << "Uploading training image set...  ";
	printPercentage(0, 0, true);

	std::vector<float> image_vec(imageSize);
	for (auto i = 0; i < numTrainingImages; ++i) {
		imageToVector<uint8_t, float>(dataset.training_images[i], image_vec);
		images.push_back(image_vec);
		printPercentage(i, numTrainingImages);
	}
	std::cout << "\nSome " << numTrainingImages << " training images have been uploaded to the GPU.\n\n";

	std::cout << "Uploading testing image set...   ";
	printPercentage(0, 0, true);
	
	for (auto i = 0; i < numTestImages; ++i) {
		imageToVector<uint8_t, float>(dataset.test_images[i], image_vec);
		testImages.push_back(image_vec);
		printPercentage(i, numTestImages);
	}
	std::cout << "\nSome " << numTestImages << " test images have been uploaded to the GPU.\n\n";


	// create a zero filled temp vector [outputSize],
	// set the element [the value of label] to 1.0,
	// convert the temp vector to a GPU matrix of the size [1, outputSize]
	// store the resulting matrix to <labels> vector of matrices
	
	std::cout << "Uploading training labels set... ";
	printPercentage(0, 0, true);
	
	std::vector<float> label_vec(outputSize);
	for (auto i = 0; i < numTrainingLabels; ++i) {
		labelToVector<uint8_t, float>(dataset.training_labels[i], outputSize, label_vec);
		labels.push_back(Matrix(label_vec));
		printPercentage(i, numTrainingLabels);
	}
	std::cout << "\nSome " << numTrainingLabels << " training labels have been uploaded to the GPU.\n\n";

	std::cout << "The amount of RAM taken:         " << makeReadable((images.size() + testImages.size() + labels.size()) * sizeof(Matrix)) << "\n";
	std::cout << "The amount of GPU memory taken:  ";
	std::cout << makeReadable((images.size() + testImages.size()) * images[0].memSize() + (labels.size() * labels[0].memSize())) << "\n\n" << std::endl;


	std::cout << "Initializing a neural network... ";
	// Init network
	std::vector<int> nnParams = { static_cast<int>(imageSize), hiddenSize, outputSize };

	NeuralNetwork nn(nnParams, 0.9f);
	std::cout << "ok\n";
	for (auto i = 0; i < nnParams.size(); ++i) {
		if (i == 0)
			std::cout << "Inputs:                          ";
		else if (i == nnParams.size() - 1)
			std::cout << "Outputs:                         ";
		else
			std::cout << "Hidden:                          ";
		std::cout << nnParams[i] << std::endl;
	}

	auto numPasses = 30;
	auto numImages = static_cast<int>(std::min(images.size(), labels.size()));

	numImages = std::min(numImages, maxImagesToProcess);

	std::cout << "\n\nTraining the network on " << numImages << " images" << std::endl;
	for (auto pass = 0; pass < numPasses; ++pass) {
		printPercentage(0, 0, true);
		std::cout << "Pass " << pass << "/" << numPasses << "... ";
		for (auto i = 0; i < numImages; ++i) {
			nn.propagateForward(images[i]);
			nn.propagateBackward(labels[i]);
			printPercentage(i, numImages);
		}
		std::cout << std::endl;
	}

	std::cout << "\nThe size of the trained network: " << makeReadable(nn.getSizeInMemory()) << "\n" << std::endl;

	const auto fname = "neuralnetwork.txt";
	std::cout << "Writing the trained network to " << fname << "... ";
	nn.saveToFile(fname);
	std::cout << "ok" << std::endl;
	
	std::vector<float> predicted(outputSize);		// predicted output
	std::vector<float> reference(outputSize);		// reference output
	std::vector<double> MSE = {};					// mean sqare error 

	std::cout << "\n\nTesting the neural network...    ";
	printPercentage(0, 0, true);

	auto hits = 0;
	auto misses = 0;

	for (auto i = 0; i < numTestImages; ++i) {
		nn.propagateForward(testImages[i]);
		nn.getCurrentOutput(predicted);
		labelToVector<uint8_t, float>(dataset.test_labels[i], outputSize, reference);
		auto meanSqError = meanSquaredError<float>(reference, predicted);
		MSE.push_back(meanSqError);

		if (isHit(predicted, static_cast<int>(dataset.test_labels[i])))
			++hits;
		else
			++misses;

		/*
		std::cout << "predicted: ";
		for (const auto elem : predicted)
			std::cout << elem << "  ";
		std::cout << std::endl;
		*/
		
		/*
		std::cout << "reference: ";
		for (const auto elem : reference)
			std::cout << elem << "  ";
		std::cout << std::endl;
		*/

		//std::cout << "MSE: " << meanSqError << "\n";
		/*
		if (++counter >= 10) {
			std::cout << std::endl;
			counter = 0;
		}
		*/
		printPercentage(i, numTestImages);
	}

	std::cout << "\n\n";
	std::cout << "The hit/miss ratio:              " << static_cast<float>(hits) / (static_cast<float>(misses) + 0.01f) << "(" << hits << "/" << misses << ")\n";
	auto average = std::accumulate(MSE.begin(), MSE.end(), 0.0f) / static_cast<float>(MSE.size());
	std::cout << "Average mean square error:       " << average << "\n";
	auto minElem = std::min_element(MSE.begin(), MSE.end());
	std::cout << "Minimum mean square error:       " << *minElem << "\n";
	auto maxElem = std::max_element(MSE.begin(), MSE.end());
	std::cout << "Maximum mean square error:       " << *maxElem << "\n" << std::endl;
	


	std::cout << "Cleaning up...                   ";
	MSE.clear();
	images.clear();
	labels.clear();
	testImages.clear();
	testLabels.clear();
	predicted.clear();
	reference.clear();
	image_vec.clear();
	label_vec.clear();
	dataset.training_images.clear();
	dataset.training_labels.clear();
	dataset.test_images.clear();
	dataset.test_labels.clear();
	std::cout << "ok" << std::endl;

#ifndef  __APPLE__
	std::cout << "Press ENTER to exit...\n";
	std::cin.get();
#endif // ! __APPLE__

	return 0;
}
