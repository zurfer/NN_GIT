//
//  random.h
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#pragma once

#include <random>

template <typename T> 
T random01() {
	static std::default_random_engine e;
	static std::uniform_real_distribution<T> dis(0, 1); 
	return dis(e);
}
