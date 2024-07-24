/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <vector>

#include "event_timer.hpp"
#include <omp.h>

using namespace std;

#include <iostream>
#include <memory>
#include <string>

#define BUFSIZE (1024 * 1024 * 80)


#define BUFSIZE_A (1024  * 1024 * 64)//1024//(1024 * 32)//
#define BUFSIZE_B (1024  * 1024 * 64)//1024//(1024 * 32)//
#define BUFSIZE_C (1024  * 1024 * 64)//1024//(1024 * 32)//
#define BUFSIZE_D (1024  * 1024 * 64)//1024//(1024 * 32)//


int krnl_M = 0;
int krnl_K = 0;
int krnl_N = 0;




int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    EventTimer et;
// 	krnl_M = 515;
//    krnl_K = 512;
//    krnl_N = 512;
// 	krnl_M = 1024;
//    krnl_K = 1024;
//    krnl_N = 8192;
    krnl_M = 2048;
    krnl_K = 4096;
    krnl_N = 16384;

    std::string binaryFile = argv[1];

    et.add("Allocate Memory in Host Memory");
    // Allocate Memory in Host Memory
    std::vector<float, aligned_allocator<float> > a(BUFSIZE_A);
    std::vector<float, aligned_allocator<float> > b(BUFSIZE_B);
    std::vector<float, aligned_allocator<float> > c(BUFSIZE_C);
    std::vector<float, aligned_allocator<float> > d(BUFSIZE_D);
    std::vector<float, aligned_allocator<float> > d2(BUFSIZE_D);
    std::vector<float, aligned_allocator<float> > d3(BUFSIZE_D);
    std::vector<float, aligned_allocator<float> > a2(BUFSIZE_A);
    std::vector<float, aligned_allocator<float> > b2(BUFSIZE_B);
    std::vector<float, aligned_allocator<float> > c2(BUFSIZE_C);
    std::vector<float, aligned_allocator<float> > a3(BUFSIZE_A);
    std::vector<float, aligned_allocator<float> > b3(BUFSIZE_B);
    std::vector<float, aligned_allocator<float> > c3(BUFSIZE_C);
    std::vector<float, aligned_allocator<float> > cfull(BUFSIZE_C);
    et.finish();

    et.add("Fill the buffers");
    // Create the test data and Software Result
    for (int i = 0; i < BUFSIZE_A; i++) {
    	if(i%3 == 0)
    	{
    		a[i] = 0.1;
    	}
    	if(i%3 == 1)
    	{
    		a[i] = 0.2;
    	}
    	if(i%3 == 2)
    	{
    		a[i] = 0.3;
    	}
    }
    for (int i = 0; i < BUFSIZE_B; i++) {
    	if(i%3 == 0)
    	{
    		b[i] = 1;
    	}
    	if(i%3 == 1)
    	{
    		b[i] = 0.5;
    	}
    	if(i%3 == 2)
    	{
    		b[i] = 0.5;
    	}
    }
    for (int i = 0; i < BUFSIZE_C; i++) {
    		c[i] = 0;
    }
    for (int i = 0; i < BUFSIZE_D; i++) {
    	d[i] = 0;
    	d2[i] = 0;
    	d3[i] = 0;
    	cfull[i]=0;
    }
    for (int i = 0; i < BUFSIZE_A; i++) {
    	a2[i] = a[i];
    	a3[i] = a[i];
    }
    for (int i = 0; i < BUFSIZE_B; i++) {
    	b2[i] = b[i];
    	b3[i] = b[i];
    }
    for (int i = 0; i < BUFSIZE_C; i++) {
    	c2[i] = c[i];
    	c3[i] = c[i];
    }

    et.finish();

    int len1= int(krnl_M/3);
    int len2= int(2*krnl_M/3);

    et.add("OpenCL host code");
    // OPENCL HOST CODE AREA START
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_vector_add1;
    cl::Kernel krnl_vector_add2;
    cl::Kernel krnl_vector_add3;
    auto devices = xcl::get_xil_devices();
    et.finish();

    et.add("Read_binary_file");
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vector_add1 = cl::Kernel(program, "wide_vadd", &err));
            OCL_CHECK(err, krnl_vector_add2 = cl::Kernel(program, "wide_vadd2", &err));
            OCL_CHECK(err, krnl_vector_add3 = cl::Kernel(program, "wide_vadd3", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    et.finish();

    et.add("Allocate Buffer in Global Memory");
    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer a_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_M * krnl_K * sizeof(float),
    								a.data(), &err));
    OCL_CHECK(err, cl::Buffer b_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_K * krnl_N * sizeof(float),
    								b.data(), &err));
    OCL_CHECK(err, cl::Buffer c_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, krnl_M * krnl_N * sizeof(float),
                              	  	c.data(), &err));
    OCL_CHECK(err, cl::Buffer d_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, krnl_M * krnl_N * sizeof(float),
                              	  	d.data(), &err));
    et.finish();

    et.add("Allocate Buffer in Global Memory");
    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer a_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_M * krnl_K * sizeof(float),
    								a2.data(), &err));
    OCL_CHECK(err, cl::Buffer b_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_K * krnl_N * sizeof(float),
    								b2.data(), &err));
    OCL_CHECK(err, cl::Buffer c_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, krnl_M * krnl_N * sizeof(float),
                              	  	c2.data(), &err));
    OCL_CHECK(err, cl::Buffer d_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, krnl_M * krnl_N * sizeof(float),
                              	  	d2.data(), &err));
    et.finish();

    et.add("Allocate Buffer in Global Memory");
    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer a_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_M * krnl_K * sizeof(float),
    								a3.data(), &err));
    OCL_CHECK(err, cl::Buffer b_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_K * krnl_N * sizeof(float),
    								b3.data(), &err));
    OCL_CHECK(err, cl::Buffer c_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, krnl_M * krnl_N * sizeof(float),
                              	  	c3.data(), &err));
    OCL_CHECK(err, cl::Buffer d_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, krnl_M * krnl_N * sizeof(float),
                              	  	d3.data(), &err));
    et.finish();

    et.add("Set the Kernel Arguments");
    // Set the Kernel Arguments
    int nargs = 0;
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, a_buf));
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, b_buf));
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, c_buf));
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, d_buf));
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, krnl_K));
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, krnl_N));
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, 0));
    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, len1));

    nargs = 0;
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, a_buf2));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, b_buf2));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, c_buf2));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, d_buf2));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, krnl_K));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, krnl_N));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, len1));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, len2));

    nargs = 0;
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, a_buf3));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, b_buf3));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, c_buf3));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, d_buf3));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, krnl_K));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, krnl_N));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, len2));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, krnl_M));
    et.finish();

    et.add("Copy input data to device global memory");
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a_buf, b_buf, c_buf, d_buf}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a_buf2, b_buf2, c_buf2, d_buf2}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a_buf3, b_buf3, c_buf3, d_buf3}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.finish());
    et.finish();


    et.add("Launch the Kernels");
    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add1));
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add2));
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add3));
    OCL_CHECK(err, err = q.finish());
    et.finish();

    et.add("Copy Result from Device Global Memory");
    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({c_buf, d_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({c_buf2, d_buf2}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({c_buf3, d_buf3}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    et.finish();

    int match = 0;
//
//
#pragma omp parallel for
    for(int m = 0; m < krnl_M; ++m){
        for(int k = 0; k < krnl_K; ++k){
			for(int j = 0; j < krnl_N; ++j)
			{
				if(m>=0 && m < krnl_M/3)
				{
					d[m*krnl_N+j]=d[m*krnl_N+j];
				}
				else if(m>=krnl_M/3 && m < 2*krnl_M/3)
				{
					d[m*krnl_N+j]=d2[m*krnl_N+j];
				}
				else if(m>=2*krnl_M/3 && m < krnl_M)
				{
					d[m*krnl_N+j]=d3[m*krnl_N+j];
				}
			}
        }
    }


    et.add("Test Implementation");
    auto start_parallel = chrono::high_resolution_clock::now();
    // Compare the results of the Device to the simulation


#pragma omp parallel for
    for(int m = 0; m < krnl_M; ++m){
        for(int k = 0; k < krnl_K; ++k){
			for(int j = 0; j < krnl_N; ++j)
			{
				cfull[m*krnl_N+j] += a[m*krnl_K+k]*b[k*krnl_N+j];
			}
        }
    }
    auto end_parallel = chrono::high_resolution_clock::now();
    et.finish();

    et.add("Find the error");
	#pragma omp parallel for
    for(int m = 0; m < krnl_M; ++m){
        for(int k = 0; k < krnl_K; ++k){
        	if(match)
        	{
        		break;
        	}
			for(int j = 0; j < krnl_N; ++j)
			{
				if(match)
				{
					break;
				}
				if (abs(d[m*krnl_N+j] - cfull[m*krnl_N+j])>1e-1)
				{
					std::cout << "ERROR: software and hardware vadd do not match: "
							<< d[m*krnl_N+j] << "!=" << cfull[m*krnl_N+j] << " at position " << m*krnl_N+j << std::endl;
					match = 1;
				}
			}
        }
    }

//    finish:
    et.finish();
	auto duration_parallel = chrono::duration_cast<chrono::milliseconds>(end_parallel - start_parallel);
	cout << "Time taken for serial matrix multiplication: " << duration_parallel.count() << " milliseconds" << endl;

	et.print();

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}



///**
//* Copyright (C) 2019-2021 Xilinx, Inc
//*
//* Licensed under the Apache License, Version 2.0 (the "License"). You may
//* not use this file except in compliance with the License. A copy of the
//* License is located at
//*
//*     http://www.apache.org/licenses/LICENSE-2.0
//*
//* Unless required by applicable law or agreed to in writing, software
//* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
//* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
//* License for the specific language governing permissions and limitations
//* under the License.
//*/
//
//#include "xcl2.hpp"
//#include <vector>
//
//#include "event_timer.hpp"
//#include <omp.h>
//
//using namespace std;
//
//#include <iostream>
//#include <memory>
//#include <string.h>
//
//#define BUFSIZE (1024 * 1024 * 80)
//
//
//#define BUFSIZE_A (1024  * 1024 * 8)//1024//(1024 * 32)//
//#define BUFSIZE_B (1024  * 1024 * 64)//1024//(1024 * 32)//
//#define BUFSIZE_C (1024  * 1024 * 32)//1024//(1024 * 32)//
//#define BUFSIZE_D (1024  * 1024 * 32)//1024//(1024 * 32)//
//
//
//int krnl_M = 0;
//int krnl_K = 0;
//int krnl_N = 0;
//
//
//
//
//int main(int argc, char** argv) {
//    if (argc != 2) {
//        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
//        return EXIT_FAILURE;
//    }
//
//    EventTimer et;
//    // 	krnl_M = 512;
//    //    krnl_K = 512;
//    //    krnl_N = 512;
//     	krnl_M = 2048;
//        krnl_K = 2048;
//        krnl_N = 4096;
//    //    krnl_M = 2048;
//    //    krnl_K = 4096;
//    //    krnl_N = 4096;
//
//    int len1= 16*(int(krnl_M/(16*3)));
//    int len2= 2*len1;
//    int start_row =0;
//
//    std::string binaryFile = argv[1];
//
//    et.add("Allocate Memory in Host Memory");
//    // Allocate Memory in Host Memory
//    std::vector<float, aligned_allocator<float> > a(len1*krnl_K);
//    std::vector<float, aligned_allocator<float> > b(krnl_K*krnl_N);
//    std::vector<float, aligned_allocator<float> > c(len1*krnl_N);
//    std::vector<float, aligned_allocator<float> > d(len1*krnl_N);
//    std::vector<float, aligned_allocator<float> > d2((len2-len1)*krnl_N);
//    std::vector<float, aligned_allocator<float> > d3((krnl_M-len2)*krnl_N);
//    std::vector<float, aligned_allocator<float> > a2((len2-len1)*krnl_K);
//    std::vector<float, aligned_allocator<float> > b2(krnl_K*krnl_N);
//    std::vector<float, aligned_allocator<float> > c2((len2-len1)*krnl_N);
//    std::vector<float, aligned_allocator<float> > a3((krnl_M-len2)*krnl_K);
//    std::vector<float, aligned_allocator<float> > b3(krnl_K*krnl_N);
//    std::vector<float, aligned_allocator<float> > c3((krnl_M-len2)*krnl_N);
//
//    std::vector<float, aligned_allocator<float> > cfull(krnl_M*krnl_N);
//    et.finish();
//
//    et.add("Fill the buffers");
//    // Create the test data and Software Result
//    for (int i = 0; i < krnl_M; i++) {
//    	for(int j = 0; j < krnl_K; j++) {
//			if(i<len1)
//			{
//				if(i%3 == 0)
//				{
//					a[i*krnl_K+j] = 0.1;
//				}
//				if(i%3 == 1)
//				{
//					a[i*krnl_K+j] = 0.2;
//				}
//				if(i%3 == 2)
//				{
//					a[i*krnl_K+j] = 0.3;
//				}
//			}
//
//			if(i>=len1 && i<len2)
//			{
//				if(i%3 == 0)
//				{
//					a2[(i-len1)*krnl_K+j] = 0.1;
//				}
//				if(i%3 == 1)
//				{
//					a2[(i-len1)*krnl_K+j] = 0.2;
//				}
//				if(i%3 == 2)
//				{
//					a2[(i-len1)*krnl_K+j] = 0.3;
//				}
//			}
//			if(i>=len2 && i <krnl_M)
//			{
//				if(i%3 == 0)
//				{
//					a3[(i-len2)*krnl_K+j] = 0.1;
//				}
//				if(i%3 == 1)
//				{
//					a3[(i-len2)*krnl_K+j] = 0.2;
//				}
//				if(i%3 == 2)
//				{
//					a3[(i-len2)*krnl_K+j] = 0.3;
//				}
//			}
//    	}
//    }
//    for (int i = 0; i < (krnl_K*krnl_N); i++) {
//    	if(i%3 == 0)
//    	{
//    		b[i] = 1;
//    	}
//    	if(i%3 == 1)
//    	{
//    		b[i] = 0.5;
//    	}
//    	if(i%3 == 2)
//    	{
//    		b[i] = 0.5;
//    	}
//    }
//    for (int i = 0; i < (krnl_K*krnl_N); i++) {
//    	b2[i] = b[i];
//    	b3[i] = b[i];
//    }
//    et.finish();
//
//    et.add("OpenCL host code");
//    // OPENCL HOST CODE AREA START
//    cl_int err;
//    cl::CommandQueue q;
//    cl::Context context;
//    cl::Kernel krnl_vector_add1;
//    cl::Kernel krnl_vector_add2;
//    cl::Kernel krnl_vector_add3;
//    auto devices = xcl::get_xil_devices();
//    et.finish();
//
//    et.add("Read_binary_file");
//    // read_binary_file() is a utility API which will load the binaryFile
//    // and will return the pointer to file buffer.
//    auto fileBuf = xcl::read_binary_file(binaryFile);
//    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
//    bool valid_device = false;
//    for (unsigned int i = 0; i < devices.size(); i++) {
//        auto device = devices[i];
//        // Creating Context and Command Queue for selected Device
//        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
//        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
//
//        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
//        cl::Program program(context, {device}, bins, nullptr, &err);
//        if (err != CL_SUCCESS) {
//            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
//        } else {
//            std::cout << "Device[" << i << "]: program successful!\n";
//            OCL_CHECK(err, krnl_vector_add1 = cl::Kernel(program, "wide_vadd", &err));
//            OCL_CHECK(err, krnl_vector_add2 = cl::Kernel(program, "wide_vadd2", &err));
//            OCL_CHECK(err, krnl_vector_add3 = cl::Kernel(program, "wide_vadd3", &err));
//            valid_device = true;
//            break; // we break because we found a valid device
//        }
//    }
//    if (!valid_device) {
//        std::cout << "Failed to program any device found, exit!\n";
//        exit(EXIT_FAILURE);
//    }
//    et.finish();
//
//    et.add("Allocate Buffer in Global Memory");
//    // Allocate Buffer in Global Memory
//    OCL_CHECK(err, cl::Buffer a_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, len1 * krnl_K * sizeof(float),
//    								a.data(), &err));
//    OCL_CHECK(err, cl::Buffer b_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_K * krnl_N * sizeof(float),
//    								b.data(), &err));
//    OCL_CHECK(err, cl::Buffer c_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, len1 * krnl_N * sizeof(float),
//                              	  	c.data(), &err));
//    OCL_CHECK(err, cl::Buffer d_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, len1 * krnl_N * sizeof(float),
//                              	  	d.data(), &err));
//    et.finish();
//
//    et.add("Allocate Buffer in Global Memory");
//    // Allocate Buffer in Global Memory
//    OCL_CHECK(err, cl::Buffer a_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (len2-len1) * krnl_K * sizeof(float),
//    								a2.data(), &err));
//    OCL_CHECK(err, cl::Buffer b_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_K * krnl_N * sizeof(float),
//    								b2.data(), &err));
//    OCL_CHECK(err, cl::Buffer c_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (len2-len1) * krnl_N * sizeof(float),
//                              	  	c2.data(), &err));
//    OCL_CHECK(err, cl::Buffer d_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, (len2-len1) * krnl_N * sizeof(float),
//                              	  	d2.data(), &err));
//    et.finish();
//
//    et.add("Allocate Buffer in Global Memory");
//    // Allocate Buffer in Global Memory
//    OCL_CHECK(err, cl::Buffer a_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (krnl_M-len2) * krnl_K * sizeof(float),
//    								a3.data(), &err));
//    OCL_CHECK(err, cl::Buffer b_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, krnl_K * krnl_N * sizeof(float),
//    								b3.data(), &err));
//    OCL_CHECK(err, cl::Buffer c_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (krnl_M-len2) * krnl_N * sizeof(float),
//                              	  	c3.data(), &err));
//    OCL_CHECK(err, cl::Buffer d_buf3(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, (krnl_M-len2) * krnl_N * sizeof(float),
//                              	  	d3.data(), &err));
//    et.finish();
//
//    et.add("Set the Kernel Arguments");
//    // Set the Kernel Arguments
//    int nargs = 0;
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, a_buf));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, b_buf));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, c_buf));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, d_buf));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, krnl_M));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, krnl_K));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, krnl_N));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, 0));
//    OCL_CHECK(err, err = krnl_vector_add1.setArg(nargs++, len1));
//
//    nargs = 0;
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, a_buf2));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, b_buf2));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, c_buf2));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, d_buf2));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, krnl_M));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, krnl_K));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, krnl_N));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, 0));
//    OCL_CHECK(err, err = krnl_vector_add2.setArg(nargs++, (len2-len1)));
//
//    nargs = 0;
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, a_buf3));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, b_buf3));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, c_buf3));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, d_buf3));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, krnl_M));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, krnl_K));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, krnl_N));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, 0));
//    OCL_CHECK(err, err = krnl_vector_add3.setArg(nargs++, (krnl_M-len2)));
//    et.finish();
//
//    et.add("Copy input data to device global memory");
//    // Copy input data to device global memory
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a_buf, b_buf, c_buf}, 0 /* 0 means from host*/));
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a_buf2, b_buf2, c_buf2}, 0 /* 0 means from host*/));
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a_buf3, b_buf3, c_buf3}, 0 /* 0 means from host*/));
//    OCL_CHECK(err, err = q.finish());
//    et.finish();
//
//
//    et.add("Launch the Kernels");
//    // Launch the Kernel
//    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add1));
//    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add2));
//    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add3));
//    OCL_CHECK(err, err = q.finish());
//    et.finish();
//
//    et.add("Copy Result from Device Global Memory");
//    // Copy Result from Device Global Memory to Host Local Memory
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_buf2}, CL_MIGRATE_MEM_OBJECT_HOST));
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_buf3}, CL_MIGRATE_MEM_OBJECT_HOST));
//    OCL_CHECK(err, err = q.finish());
//    et.finish();
//
//    int match = 0;
//
////
////#pragma omp parallel for
////    for(int m = 0; m < krnl_M; ++m){
////        for(int k = 0; k < krnl_K; ++k){
////			for(int j = 0; j < krnl_N; ++j)
////			{
////				if(m>=0 && m < len1)
////				{
////					d[m*krnl_N+j]=d[m*krnl_N+j];
////				}
////				else if(m>=len1 && m < len2)
////				{
////					d[m*krnl_N+j]=d2[(m-len1)*krnl_N+j];
////				}
////				else if(m>=len2 && m < krnl_M)
////				{
////					d[m*krnl_N+j]=d3[(m-len2)*krnl_N+j];
////				}
////			}
////        }
////    }
////
////
//    et.add("Test Implementation");
////    auto start_parallel = chrono::high_resolution_clock::now();
////    // Compare the results of the Device to the simulation
////
////
//#pragma omp parallel for
//    for(int m = 0; m < krnl_M; ++m){
//        for(int k = 0; k < krnl_K; ++k){
//			for(int j = 0; j < krnl_N; ++j)
//			{
//				if(m<len1)
//				{
//					cfull[m*krnl_N+j] += a[m*krnl_K+k]*b[k*krnl_N+j];
//				}
//
//				if(m>=len1 && m<len2)
//				{
//					cfull[m*krnl_N+j] += a2[(m-len1)*krnl_K+k]*b[k*krnl_N+j];
//				}
//				if(m>=len2 && m <krnl_M)
//				{
//					cfull[m*krnl_N+j] += a3[(m-len2)*krnl_K+k]*b[k*krnl_N+j];
//				}
//			}
//        }
//    }
////    auto end_parallel = chrono::high_resolution_clock::now();
//    et.finish();
////
//    et.add("Find the error");
////	#pragma omp parallel for
//    for(int m = 0; m < krnl_M; ++m){
//        for(int k = 0; k < krnl_K; ++k){
//
//			for(int j = 0; j < krnl_N; ++j)
//			{
//				if(!match)
//				{
//					if(m<len1)
//					{
//						if (d[m*krnl_N+j] != cfull[m*krnl_N+j])
//						{
//							std::cout << "ERROR: software and hardware vadd do not match: "
//									<< d[m*krnl_N+m] << "!=" << a[m*krnl_K+k]*b[k*krnl_N+j] << " at position " << m*krnl_N+j
//									<< " m is " << m << " n is " << j << std::endl;
//							match = 1;
//						}
//					}
//
//					if(m>=len1 && m<len2)
//					{
//						if (d2[(m-len1)*krnl_N+j] != cfull[m*krnl_N+j])
//						{
//							std::cout << "ERROR: software and hardware vadd do not match: "
//									<< d2[(m-len1)*krnl_N+m] << "!=" << a2[(m-len1)*krnl_K+k]*b[k*krnl_N+j] << " at position " << m*krnl_N+j
//									<<" m is " << m << " n is " << j << std::endl;
//							match = 1;
//						}
//					}
//					if(m>=len2 && m <krnl_M)
//					{
//						if (d3[(m-len2)*krnl_N+j] != cfull[m*krnl_N+j])
//						{
//							std::cout << "ERROR: software and hardware vadd do not match: "
//									<< d3[(m-len2)*krnl_N+m] << "!=" << a3[(m-len2)*krnl_K+k]*b[k*krnl_N+j] << " at position " << m*krnl_N+j
//									<< " m is " << m << " n is " << j << std::endl;
//							match = 1;
//						}
//					}
//			}
//			}
//        }
//    }
//    et.finish();
////
////	auto duration_parallel = chrono::duration_cast<chrono::milliseconds>(end_parallel - start_parallel);
////	cout << "Time taken for serial matrix multiplication: " << duration_parallel.count() << " milliseconds" << endl;
//
//	et.print();
//
//    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
//    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
//}
//
//
