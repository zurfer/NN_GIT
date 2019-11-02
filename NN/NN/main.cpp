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
	
    std::cout << "m1:\n";
	Matrix<double> m1(3, 4, {
			1.0, 2.0, 3.0, 100.0,
			4.0, 5.0, 6.0, 100.0,
			7.0, 8.0, 9.0, 100.0});
    m1.print();
    
    std::cout << "m2 {m1} \n";
    Matrix<double> m2 {m1};
    m2.print();
    
    std::cout << "m2.randomize()\n";
    m2.randomize();
    m2.print();
    
    std::cout << "m2 = m1\n";
    m2 = m1;
    m2.print();
    
    Matrix<float> m3;
    std::cout << "m3; m3.conv_from m1";
    m3 = m1;
    m1.print();
    

	//std::cout << "Press ENTER to exit...\n";
	//std::cin.get();

    return 0;
}
