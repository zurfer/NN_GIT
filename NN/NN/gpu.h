//
//  gpu.h
//  NN
//
//  Created by Stanislav Anisimov on 29.10.2019.
//  Copyright © 2019 Stanislav Anisimov. All rights reserved.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <boost/compute.hpp>
#include "utils.h"

namespace compute = boost::compute;
using compute::uint_;

namespace GPU {

	// The GPU context class
	class Device {
	public:
		Device()
			: device(compute::system::default_device()),
			context(device), queue(context, device),
			random_engine(queue), rnd_dist(-1.0f, 1.00001f),
			num_compute_units{ device.compute_units() }
		{
		}

		// prints device info
		void print_info() {
			std::cout << "Device name:                     " << device.name() << "\n";
			std::cout << "Total memory:                    " << makeReadable(device.global_memory_size()) << " (" << device.global_memory_size() << ")" << "\n";
			std::cout << "Max memory per allocation:       " << makeReadable(device.max_memory_alloc_size()) << " (" << device.max_memory_alloc_size() << ")" << "\n";
			std::cout << "Max workgroup size:              " << device.max_work_group_size() << "\n";
			std::cout << "Number of compute units:         " << num_compute_units << "\n" << std::endl;
		}

		const compute::device device;								// computing device (gpu or a fallback to cpu)
		compute::context context;									// computing context
		compute::command_queue queue;								// main queue
		compute::default_random_engine random_engine;				// rnd engine
		compute::uniform_real_distribution<float> rnd_dist;			// rnd generator
		uint_ num_compute_units;									// number of compute units
	};
	
	static Device& defGPU() {
		static Device gpu = {};
		
		return gpu;
	}

} // the end of the GPU namespace

