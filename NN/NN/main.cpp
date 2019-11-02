//
//  main.cpp
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#include <iostream>
#include "Matrix.h"

int main(int argc, const char * argv[]) {
    
	//srand(time(nullptr));
	
	Matrix<double> m1(3, 4, {
			1.0, 2.0, 3.0, 100.0,
			4.0, 5.0, 6.0, 100.0,
			7.0, 8.0, 9.0, 100.0,
			10.0, 11.0, 12.0, 100.0});
	
	Matrix<float> m2(4, 4);

	Matrix<float> m3;

    std::cout << "m1:\n";
	m1.print();

	std::cout << "m2:\n";
	m2.print();
	
	std::cout << "m3:\n";
	m3.print();

	m3.convert_from(m1);
	std::cout << "convert m1 to m3 and print m3:\n";
	m3.print();

	m3.randomize();
	std::cout << "randomize and print m3:\n";
	m3.print();

	std::cout << "Press ENTER to exit...\n";
	std::cin.get();

    return 0;
}
