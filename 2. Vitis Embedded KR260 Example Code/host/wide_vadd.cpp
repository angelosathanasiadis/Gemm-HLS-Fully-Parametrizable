

#include "event_timer.hpp"

#include <iostream>
#include <memory>
#include <string>

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl.hpp"

#ifdef HW_EMU
#define BUFSIZE (1024 * 32)
#else
#define BUFSIZE (1024 * 1024 * 32)
#endif

#define BUFSIZE_A (1024  * 1024 * 34)//1024//(1024 * 32)//
#define BUFSIZE_B (1024  * 1024 * 66)//1024//(1024 * 32)//
#define BUFSIZE_C (1024  * 1024 * 10)//1024//(1024 * 32)//
#define BUFSIZE_D (1024  * 1024 * 10)//1024//(1024 * 32)//


int krnl_M = 2048;
int krnl_K = 4096;
int krnl_N = 16348;


void test_kernel_big(int M, int N, int K,
        float *A,
        float *B,
        float *C,
		float *D)
{
	 bool verified = true;
	 int int_counter=0;

    for(int m = 0; m < M; ++m){
        for(int k = 0; k < K; ++k){
			for(int j = 0; j < N; ++j)
			{
				C[m*N+j] += A[m*K+k]*B[k*N+j];

			}
        }
    }
    for(int m = 0; m < M; ++m){
        for(int k = 0; k < K; ++k){
			for(int j = 0; j < N; ++j){
				if (D[m*N+j] != C[m*N+j])
				{
					verified = false;
					int_counter++;
					std::cout << "ERROR: software and hardware vadd do not match: "
							<< D[m*N+j] << "!=" << C[m*N+j] << " at position " << m*N+j << std::endl;

					if (verified) {
						std::cout
							<< std::endl
							<< "OCL-mapped contiguous buffer example complete!"
							<< std::endl
							<< std::endl;
					}
					else {
						std::cout
							<< std::endl
							<< "OCL-mapped contiguous buffer example complete! (with errors)"
							<< std::endl
							<< std::endl;
					}
					return;
				}
			}
        }
    }

	if (verified) {
	    std::cout
	        << std::endl
	        << "OCL-mapped contiguous buffer example complete!"
	        << std::endl
	        << std::endl;
	}
	else {
	    std::cout
	        << std::endl
	        << "OCL-mapped contiguous buffer example complete! (with errors)" << int_counter
	        << std::endl
	        << std::endl;
	}
}





int main(int argc, char *argv[])
{
//    krnl_M = 1024;
//    krnl_K = 1024;
//    krnl_N = 1024;
    krnl_M = 2048;
    krnl_K = 16384;
    krnl_N = 4096;

    std::cout << "--------------- Start ---------------" << std::endl;


    // Initialize an event timer we'll use for monitoring the application
    EventTimer et;
    EventTimer ett;

    ett.add("Program");
    std::cout << "-- Parallelizing the Data Path --" << std::endl
              << std::endl;

    // Initialize the runtime (including a command queue) and load the
    // FPGA image
    std::cout << "Loading binary_container_1.xclbin to program the board" << std::endl
              << std::endl;
    et.add("OpenCL Initialization");

    // This application will use the first Xilinx device found in the system
    swm::XilinxOcl xocl;
    xocl.initialize("binary_container_1.xclbin");

    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl    = xocl.get_kernel("wide_vadd");
    et.finish();

    /// New code for example 01
    std::cout << "Running kernel test XRT-allocated buffers and wide data path:" << std::endl
              << std::endl;

    // Map our user-allocated buffers as OpenCL buffers using a shared
    // host pointer
    et.add("Allocate contiguous OpenCL buffers");

	 cl_mem_ext_ptr_t bank_ext;
	 bank_ext.flags = 2 | XCL_MEM_TOPOLOGY;
	 bank_ext.obj   = NULL;
	 bank_ext.param = 0;
	 cl::Buffer a_buf(xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					  krnl_M * krnl_K * sizeof(float),
					  NULL,
					  NULL);
	 cl::Buffer b_buf(xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					  krnl_K * krnl_N * sizeof(float),
					  NULL,
					  NULL);
	 cl::Buffer c_buf(xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					  krnl_M * krnl_N * sizeof(float),
					  NULL,
					  NULL);
	 cl::Buffer d_buf(xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_WRITE),
					  krnl_M * krnl_N * sizeof(float),
					  NULL,
					  NULL);
    et.finish();


	float *a =(float *)malloc(krnl_M * krnl_K*sizeof(float));
	float *b =(float *)malloc(krnl_K * krnl_N *sizeof(float));
	float *c =(float *)malloc(krnl_M * krnl_N*sizeof(float));
	float *d =(float *)malloc(krnl_M * krnl_N*sizeof(float));

	std::cout << "Populating buffer inputs" << std::endl;
    et.add("Populating buffer inputs");

    for (int i = 0; i < krnl_M*krnl_K; i++) {
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
    for (int i = 0; i < krnl_K*krnl_N; i++) {
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
    for (int i = 0; i < krnl_M*krnl_N; i++) {
    	if(i%3 == 0)
    	{
    		c[i] = 0.2;
    	}
    	if(i%3 == 1)
    	{
    		c[i] = 0.1;
    	}
    	if(i%3 == 2)
    	{
    		c[i] = 1;
    	}
    }
    for (int i = 0; i < krnl_M*krnl_N; i++) {
    	if(i%3 == 0)
    	{
    		d[i] = 0.2;
    	}
    	if(i%3 == 1)
    	{
    		d[i] = 0.1;
    	}
    	if(i%3 == 2)
    	{
    		d[i] = 1;
    	}
    }

    et.finish();




    et.add("Map buffers to userspace pointers");
     a = (float *)q.enqueueMapBuffer(a_buf,
                                                 CL_TRUE,
                                                 CL_MAP_WRITE,
                                                 0,
												 krnl_M * krnl_K * sizeof(float));
     b = (float *)q.enqueueMapBuffer(b_buf,
                                                 CL_TRUE,
                                                 CL_MAP_WRITE,
                                                 0,
												 krnl_K * krnl_N * sizeof(float));


     c = (float *)q.enqueueMapBuffer(c_buf,
                                                 CL_TRUE,
												 CL_MAP_WRITE  | CL_MAP_READ,
                                                 0,
												 krnl_M * krnl_N * sizeof(float));

     d = (float *)q.enqueueMapBuffer(d_buf,
                                                 CL_TRUE,
                                                  CL_MAP_READ,
                                                 0,
												 krnl_M * krnl_N * sizeof(float));

    et.finish();
int startRow=0;
    // Set vadd kernel arguments. We do this before mapping the buffers to allow XRT
    // to allocate the buffers in the appropriate memory banks for the selected
    // kernels. For buffer 'd' we explicitly set a bank above, but this buffer is
    // never migrated to the Alveo card so this mapping is theoretical.
    et.add("Set kernel arguments");
    krnl.setArg(0, a_buf);
    krnl.setArg(1, b_buf);
    krnl.setArg(2, c_buf);
    krnl.setArg(3, d_buf);
    krnl.setArg(4, krnl_K);
    krnl.setArg(5, krnl_N);
    krnl.setArg(6, startRow);
    krnl.setArg(7, krnl_M);
    et.finish();

    // Send the buffers down to the Alveo card
    et.add("Memory object migration enqueue");
    cl::Event event_sp;
    q.enqueueMigrateMemObjects({a_buf, b_buf, c_buf, d_buf}, 0, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event *)&event_sp);
    et.finish();
    et.add("Kernel");

    q.enqueueTask(krnl, NULL, &event_sp);

    clWaitForEvents(1, (const cl_event *)&event_sp);



    q.enqueueMigrateMemObjects({d_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    et.finish();
//    std::cout << "Start Serial" << std::endl;
//    et.add("test Kernel");
//    test_kernel_big(krnl_M, krnl_N, krnl_K, a, b, c, d);
//	et.finish();


    std::cout << "--------------- Key execution times ---------------" << std::endl;


    q.enqueueUnmapMemObject(a_buf, a);
    q.enqueueUnmapMemObject(b_buf, b);
    q.enqueueUnmapMemObject(c_buf, c);
    q.enqueueUnmapMemObject(d_buf, d);

    ett.finish();
    et.print();
    ett.print();
    q.finish();
}
