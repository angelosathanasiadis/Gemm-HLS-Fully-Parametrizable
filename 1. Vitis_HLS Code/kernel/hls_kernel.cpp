// Including to use ap_uint<> datatype
#include <ap_int.h>
#include <stdio.h>
#include <string.h>
#include <hls_stream.h>

// Constants defining buffer and data sizes
#define BUFFER_SIZE 512
#define DATAWIDTH 512
#define DATATYPE_SIZE 32
#define VECTOR_SIZE (DATAWIDTH / DATATYPE_SIZE) // vector size is 16 (512/32 = 16)
typedef ap_uint<DATAWIDTH> uint512_dt;
typedef ap_uint<DATATYPE_SIZE> din_type;
typedef ap_uint<DATATYPE_SIZE + 1> dout_type;

// Constants for matrix sizes and unrolling factors
#define ACCESS_K 256
#define ACCESS_N 256

#define UNROLL_N 3
#define UNROLL_K 16

#define DEFstartRow 0
#define DEFendRow 2048
#define DEFsizeK 4096
#define DEFsizeN 16384

// Helper function to convert unsigned integer to float
float uint_to_float(din_type x) {
#pragma HLS INLINE
    union {
        unsigned int i;
        float f;
    } conv;
    conv.i = (unsigned int)x;
    return conv.f;
}

// Helper function to convert float to unsigned integer
unsigned int float_to_uint(float x) {
#pragma HLS INLINE
    union {
        unsigned int i;
        float f;
    } conv;
    conv.f = x;
    return conv.i;
}

// Function to read matrix A and stream it
void read_A(hls::stream<uint512_dt> &A_Stream, const uint512_dt *in, const int startRow,
            const int endRow, const int curKsz, const int k_index, const int sizeK) {
    vA_rd:
    for(int m = startRow; m < endRow; ++m) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
        #pragma HLS LOOP_FLATTEN off
        for(int k = 0; k < curKsz / VECTOR_SIZE; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=curKsz
            #pragma HLS LOOP_FLATTEN
            #pragma HLS pipeline II=1

            int index = int(m * (sizeK / VECTOR_SIZE) + k_index / VECTOR_SIZE + k);
            uint512_dt A_curr = in[index];
            A_Stream << A_curr;
        }
    }
}

// Function to read matrix C and stream it
void read_C(const uint512_dt *in, hls::stream<uint512_dt> &C_in_Stream, const int startRow,
            const int endRow, const int curNsz, const int n_index, const int sizeN) {
    vC_rd:
    for(int m = startRow; m < endRow; ++m) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
        for(int j = 0; j < curNsz; j++) {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
            #pragma HLS LOOP_FLATTEN
            #pragma HLS pipeline II=1

            int index = int((m * sizeN) / VECTOR_SIZE) + n_index + j;
            uint512_dt C_curr = in[index];
            C_in_Stream << C_curr;
        }
    }
}

// Compute function performing matrix multiplication and addition
void compute(hls::stream<uint512_dt> &A_Stream, const uint512_dt (*vB_local)[ACCESS_N], uint512_dt *vC_local, hls::stream<uint512_dt> &C_out_Stream,
             const int startRow, const int endRow, const int curKsz, const int curNsz, const int sizeN) {
    for(int m = startRow; m < endRow; ++m) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
        #pragma HLS LOOP_FLATTEN off
        // memset(vC_local,0,ACCESS_N*sizeof(uint512_dt));

        v2_rd_add:
        for(int k = 0; k < curKsz / VECTOR_SIZE; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=curKsz
            #pragma HLS LOOP_FLATTEN off
            uint512_dt tmpV3 = A_Stream.read();
            for(int vector_k = 0; vector_k < VECTOR_SIZE; vector_k += UNROLL_K) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                #pragma HLS LOOP_FLATTEN off
                for(int jj = 0; jj < curNsz; jj += UNROLL_N) {
                    #pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
                    #pragma HLS pipeline
                    for(int kk = 0; kk < UNROLL_K; kk++) {
                        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 8
                        for(int j = 0; j < UNROLL_N; j++) {
                            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 8
                            uint512_dt tmpV1 = vB_local[k * VECTOR_SIZE + kk + vector_k][jj + j];
                            uint512_dt tmpV2 = ((k + kk + vector_k) != 0) ? vC_local[jj + j] : static_cast<uint512_dt>(0);

                            uint512_dt tmpOut = 0;
                            din_type val1, val2;
                            din_type val3 = tmpV3.range(DATATYPE_SIZE * ((vector_k + kk) + 1) - 1, (vector_k + kk) * DATATYPE_SIZE);
                            dout_type res;

                            v2_parallel_add:
                            for(int vector = 0; vector < VECTOR_SIZE; vector++) {
                                #pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
                                #pragma HLS UNROLL
                                val1 = tmpV1.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
                                val2 = tmpV2.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
                                res = float_to_uint((uint_to_float(val2)) + ((uint_to_float(val3)) * (uint_to_float(val1))));
                                tmpOut.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = res;
                            }
                            vC_local[jj + j] = tmpOut;
                        }
                    }
                }
            }
        }
        vC_write_Stream:
        for(int jj = 0; jj < curNsz; jj++) {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
            #pragma HLS pipeline II=1
            C_out_Stream << vC_local[jj];
        }
    }
}

// Function to write results back to global memory
void write_C(uint512_dt *C_out, hls::stream<uint512_dt> &C_in_Stream, hls::stream<uint512_dt> &C_out_Stream, const int startRow,
             const int endRow, const int curNsz, const int n_index, const int sizeN) {
    vC_rd:
    for(int m = startRow; m < endRow; ++m) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
        for(int j = 0; j < curNsz; j++) {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
            #pragma HLS LOOP_FLATTEN
            #pragma HLS pipeline II=1

            int index = int((m * sizeN) / VECTOR_SIZE) + n_index + j;

            uint512_dt tmpV1 = C_out_Stream.read();
            uint512_dt tmpV2 = C_in_Stream.read();
            uint512_dt tmpOut = 0;
            din_type val1, val2;
            dout_type res;

            vC_parallel_add:
            for(int vector = 0; vector < VECTOR_SIZE; vector++) {
                #pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
                #pragma HLS UNROLL
                val1 = tmpV1.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
                val2 = tmpV2.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
                res = float_to_uint((uint_to_float(val2)) + (uint_to_float(val1)));
                tmpOut.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = res;
            }

            C_out[index] = tmpOut;
        }
    }
}

// Dataflow region coordinating the various read, compute, and write functions
void dataflow_region(hls::stream<uint512_dt> &A_Stream, const uint512_dt (*vB_local)[ACCESS_N], uint512_dt *vC_local, const uint512_dt *in1,
                     const uint512_dt *in3, uint512_dt *C_out, hls::stream<uint512_dt> &C_in_Stream, hls::stream<uint512_dt> &C_out_Stream, const int startRow,
                     const int endRow, const int curKsz, const int curNsz, const int k_index, const int n_index, const int sizeN, const int sizeK) {
    #pragma HLS DATAFLOW
    read_A(A_Stream, in1, startRow, endRow, curKsz, k_index, sizeK);
    read_C(in3, C_in_Stream, startRow, endRow, curNsz, n_index, sizeN);
    compute(A_Stream, vB_local, vC_local, C_out_Stream, startRow, endRow, curKsz, curNsz, sizeN);
    write_C(C_out, C_in_Stream, C_out_Stream, startRow, endRow, curNsz, n_index, sizeN);
}

// Main kernel function implementing vector addition
extern "C"
{
void wide_vadd(
    const uint512_dt *in1, // Read-Only Vector 1 (Matrix A)
    const uint512_dt *in2, // Read-Only Vector 2 (Matrix B)
    uint512_dt *in3, // Read-Only Vector 3 (Matrix C)
    uint512_dt *out, // Output Result
    int sizeK, // Number of columns in matrix A and number of rows in matrix B
    int sizeN, // Number of columns in matrix B and matrix C
    int startRow, // Starting row for processing
    int endRow // Ending row for processing
    ) {
    #pragma HLS INTERFACE m_axi port = in1 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem
    #pragma HLS INTERFACE m_axi port = in2 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem1
    #pragma HLS INTERFACE m_axi port = in3 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem2
    #pragma HLS INTERFACE m_axi port = out max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem3
    #pragma HLS INTERFACE s_axilite port = in1 bundle = control
    #pragma HLS INTERFACE s_axilite port = in2 bundle = control
    #pragma HLS INTERFACE s_axilite port = in3 bundle = control
    #pragma HLS INTERFACE s_axilite port = out bundle = control
    #pragma HLS INTERFACE s_axilite port = sizeK bundle = control
    #pragma HLS INTERFACE s_axilite port = sizeN bundle = control
    #pragma HLS INTERFACE s_axilite port = startRow bundle = control
    #pragma HLS INTERFACE s_axilite port = endRow bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    // Local memory for storing intermediate results
    uint512_dt vC_local[ACCESS_N];
    #pragma HLS BIND_STORAGE variable=vC_local type=RAM_2P impl=BRAM
    uint512_dt vB_local[ACCESS_K][ACCESS_N];
    #pragma HLS BIND_STORAGE variable=vB_local type=RAM_2P impl=BRAM

    #pragma HLS array_partition variable=vB_local type=cyclic factor=UNROLL_N dim=2
    #pragma HLS array_partition variable=vB_local type=cyclic factor=UNROLL_K dim=1
    #pragma HLS array_partition variable=vC_local type=cyclic factor=UNROLL_N

    // Streams for data transfer between functions
    static hls::stream<uint512_dt> A_Stream;
    static hls::stream<uint512_dt> C_Stream_out;
    static hls::stream<uint512_dt> C_Stream_in;

    #pragma HLS STREAM variable = A_Stream depth = 2
    #pragma HLS STREAM variable = C_Stream_in depth = 3
    #pragma HLS STREAM variable = C_Stream_out depth = 2

    // Setting default values for rows and matrix sizes
    startRow = DEFstartRow;
    endRow = DEFendRow;
    sizeK = DEFsizeK;
    sizeN = DEFsizeN;

    // Input vector size for integer vectors. 
    int sizeN_in16 = sizeN / VECTOR_SIZE; // 2048

    // Loop over the columns of matrix B in chunks of ACCESS_K
    for(int ex_k = 0; ex_k < sizeK; ex_k += ACCESS_K) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = sizeK
        #pragma HLS LOOP_FLATTEN off

        int curKsz = ACCESS_K;

        // Loop over the rows of matrix B in chunks of ACCESS_N
        for(int ex_j = 0; ex_j < sizeN_in16; ex_j += ACCESS_N) {
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = sizeN_in16
            #pragma HLS LOOP_FLATTEN off
            int curNsz = ACCESS_N;

            // Read chunk of matrix B into local memory
            vB_rd:
            for(int k = 0; k < curKsz; k++) {
                #pragma HLS LOOP_TRIPCOUNT min = 1 max = curKsz
                for(int j = 0; j < curNsz; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
                    #pragma HLS pipeline
                    vB_local[k][j] = in2[(ex_k + k) * sizeN_in16 + ex_j + j];
                }
            }

            // Alternate between reading from in3 and writing to out
            if((ex_k / ACCESS_K) % 2) {
                dataflow_region(A_Stream, vB_local, vC_local, in1, in3, out, C_Stream_in, C_Stream_out, startRow, endRow, curKsz, curNsz, ex_k, ex_j, sizeN, sizeK);
            } else {
                dataflow_region(A_Stream, vB_local, vC_local, in1, out, in3, C_Stream_in, C_Stream_out, startRow, endRow, curKsz, curNsz, ex_k, ex_j, sizeN, sizeK);
            }
        }
    }
}

}

//Including to use ap_uint<> datatype
#include <ap_int.h>
#include <stdio.h>
#include <string.h>
#include <hls_stream.h>

#define BUFFER_SIZE 512
#define DATAWIDTH 512
#define DATATYPE_SIZE 32
#define VECTOR_SIZE (DATAWIDTH / DATATYPE_SIZE) // vector size is 16 (512/32 = 16)
typedef ap_uint<DATAWIDTH> uint512_dt;
typedef ap_uint<DATATYPE_SIZE> din_type;
typedef ap_uint<DATATYPE_SIZE + 1> dout_type;


#define ACCESS_K 128
#define ACCESS_N 128

#define UNROLL_N 8
#define UNROLL_K 2


#define DEFstartRow	0
#define DEFendRow   2048
#define DEFsizeK 	4096
#define DEFsizeN	16384


float uint_to_float(din_type x) {
#pragma HLS INLINE
    union {
        unsigned int i;
        float f;
    } conv;
    conv.i=(unsigned int)x;
    return conv.f;
}

unsigned int float_to_uint(float x) {
#pragma HLS INLINE
    union {
        unsigned int i;
        float f;
    } conv;
    conv.f=x;
    return conv.i;
}


void read_A(hls::stream<uint512_dt> &A_Stream, const uint512_dt *in,  const int startRow,
		const int endRow, const int curKsz, const int k_index, const int sizeK)
{
	vA_rd:
	for(int m = startRow; m < endRow; ++m){
	#pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
	#pragma HLS LOOP_FLATTEN off
		for(int k = 0; k < curKsz/VECTOR_SIZE; k++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=curKsz
		#pragma HLS LOOP_FLATTEN
		#pragma HLS pipeline II=1

			int index = int(m*(sizeK/VECTOR_SIZE)+k_index/VECTOR_SIZE+k);
			uint512_dt A_curr = in[index];
			A_Stream << A_curr;
		}
	}
}


void read_C(const uint512_dt *in, hls::stream<uint512_dt> &C_in_Stream,  const int startRow,
		const int endRow, const int curNsz, const int n_index, const int sizeN)
{
	vC_rd:
	for(int m = startRow; m < endRow; ++m){
	#pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
		for (int j = 0; j < curNsz; j++) {
		#pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
		#pragma HLS LOOP_FLATTEN
		#pragma HLS pipeline II=1

			int index = int((m * sizeN)/VECTOR_SIZE)+n_index+j;
			uint512_dt C_curr = in[index];
			C_in_Stream << C_curr;
		}
	}
}




void compute(hls::stream<uint512_dt> &A_Stream, const uint512_dt (*vB_local)[ACCESS_N], uint512_dt *vC_local, hls::stream<uint512_dt> &C_out_Stream,
			const int startRow, const int endRow, const int curKsz, const int curNsz, const int sizeN, float in_alpha)
{
	for(int m = startRow; m < endRow; ++m){
	#pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
	#pragma HLS LOOP_FLATTEN off

		v2_rd_add:
		for(int k = 0; k < curKsz/VECTOR_SIZE; k++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=curKsz
		#pragma HLS LOOP_FLATTEN off
			uint512_dt tmpV3 = A_Stream.read();
			for (int vector_k = 0; vector_k < VECTOR_SIZE; vector_k+=UNROLL_K) {
			#pragma HLS LOOP_TRIPCOUNT min=1 max=16
			#pragma HLS LOOP_FLATTEN off
				for (int jj = 0; jj < curNsz; jj+=UNROLL_N) {
				#pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
				#pragma HLS pipeline
					for (int kk = 0; kk < UNROLL_K; kk++) {
					#pragma HLS LOOP_TRIPCOUNT min = 1 max = 8
						for (int j = 0; j < UNROLL_N; j++) {
						#pragma HLS LOOP_TRIPCOUNT min = 1 max = 8
							uint512_dt tmpV1 = vB_local[k*VECTOR_SIZE+kk+vector_k][jj+j];
							uint512_dt tmpV2 = ((k+kk+vector_k)!=0) ? vC_local[jj+j] : static_cast<uint512_dt>(0);


							uint512_dt tmpOut = 0;
							din_type val1, val2;
							din_type val3 = tmpV3.range(DATATYPE_SIZE * ((vector_k+kk) + 1) - 1, (vector_k+kk)* DATATYPE_SIZE);
							dout_type res;

							v2_parallel_add:
							for (int vector = 0; vector < VECTOR_SIZE; vector++) {
							#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
							#pragma HLS UNROLL
								val1 = tmpV1.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
								val2 = tmpV2.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
								res = float_to_uint((uint_to_float(val2))  + ((in_alpha*uint_to_float(val3))*(uint_to_float(val1))));
								tmpOut.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = res;
							}
							vC_local[jj+j] = tmpOut;
						}
					}
				}
			}
		}
		vC_write_Stream:
		for (int jj = 0; jj < curNsz; jj++){
#pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
#pragma HLS pipeline II=1
			C_out_Stream<<vC_local[jj];
		}

	}
}

void write_C(uint512_dt *C_out, hls::stream<uint512_dt> &C_in_Stream, hls::stream<uint512_dt> &C_out_Stream,const int startRow,
			const int endRow, const int curNsz, const int n_index, const int sizeN, const int k_index, float in_beta)
{
	vC_rd:
	for(int m = startRow; m < endRow; ++m){
	#pragma HLS LOOP_TRIPCOUNT min = 1 max = endRow
		for (int j = 0; j < curNsz; j++) {
		#pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
		#pragma HLS LOOP_FLATTEN
		#pragma HLS pipeline II=1

			int index = int((m * sizeN)/VECTOR_SIZE)+n_index+j;
			float temp_beta = (k_index!=0) ? in_beta : static_cast<float>(1);

			uint512_dt tmpV1 = C_out_Stream.read();
			uint512_dt tmpV2 = C_in_Stream.read();
			uint512_dt tmpOut = 0;
			din_type val1, val2;
			dout_type res;

			vC_parallel_add:
			for (int vector = 0; vector < VECTOR_SIZE; vector++) {
			#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
			#pragma HLS UNROLL
				val1 = tmpV1.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
				val2 = tmpV2.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
				res = float_to_uint((temp_beta*uint_to_float(val2))  + (uint_to_float(val1)));
				tmpOut.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = res;
			}

			C_out[index] = tmpOut;
		}
	}
}

void dataflow_region(hls::stream<uint512_dt> &A_Stream, const uint512_dt (*vB_local)[ACCESS_N], uint512_dt *vC_local, const uint512_dt *in1,
		const uint512_dt *in3, uint512_dt *C_out, hls::stream<uint512_dt> &C_in_Stream, hls::stream<uint512_dt> &C_out_Stream, const int startRow,
		const int endRow, const int curKsz, const int curNsz, const int k_index, const int n_index, const int sizeN, const int sizeK,
		float in_alpha, float in_beta)
{
	#pragma HLS DATAFLOW
	read_A(A_Stream, in1, startRow, endRow, curKsz, k_index, sizeK);
	read_C(in3, C_in_Stream, startRow, endRow, curNsz, n_index, sizeN);
	compute(A_Stream, vB_local, vC_local, C_out_Stream, startRow, endRow, curKsz, curNsz, sizeN, in_alpha);
	write_C(C_out,  C_in_Stream, C_out_Stream, startRow, endRow, curNsz, n_index, sizeN, k_index, in_beta);
}



extern "C"
{
    void wide_vadd(
        const  uint512_dt *in1, // Read-Only Vector 1
        const uint512_dt *in2, // Read-Only Vector 2
		uint512_dt *in3,		// Output Result
		uint512_dt *out,		// Output Result
		float alpha_const,
		int startRow,
		int endRow,               // Size in integer
		int sizeK,
		int sizeN,
		float beta
    )
    {
#pragma HLS INTERFACE m_axi port = in1 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = in2 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = in3 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = out max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = in2 bundle = control
#pragma HLS INTERFACE s_axilite port = in3 bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = alpha_const bundle = control
#pragma HLS INTERFACE s_axilite port = startRow bundle = control
#pragma HLS INTERFACE s_axilite port = endRow bundle = control
#pragma HLS INTERFACE s_axilite port = sizeK bundle = control
#pragma HLS INTERFACE s_axilite port = sizeN bundle = control
#pragma HLS INTERFACE s_axilite port = beta bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    	uint512_dt vC_local[ACCESS_N]; // Local memory to store vector1
//    	#pragma HLS BIND_STORAGE variable=vC_local type=RAM_2P impl=BRAM
    	uint512_dt vB_local[ACCESS_K][ACCESS_N]; // Local memory to store vector1
//    	#pragma HLS BIND_STORAGE variable=vB_local type=RAM_2P impl=BRAM

    		#pragma HLS array_partition variable=vB_local type=cyclic factor=2 dim=2//type=cyclic
    		#pragma HLS array_partition variable=vB_local type=cyclic factor=8 dim=1//type=cyclic
    		#pragma HLS array_partition variable=vC_local type=cyclic factor=8

        	static hls::stream<uint512_dt>      A_Stream;
        	static hls::stream<uint512_dt>      C_Stream_out;
        	static hls::stream<uint512_dt>      C_Stream_in;

    		#pragma HLS STREAM variable = A_Stream    depth = 2
        	#pragma HLS STREAM variable = C_Stream_in    depth = 3
        	#pragma HLS STREAM variable = C_Stream_out    depth = 2

        	startRow = DEFstartRow;
        	endRow = DEFendRow;
        	sizeK = DEFsizeK;
        	sizeN = DEFsizeN;

		int sizeN_in16 = sizeN / VECTOR_SIZE; //2048(sizeN - 1) / VECTOR_SIZE + 1; //2048
		float in_beta=beta;
		float in_alpha_const = alpha_const;

		for(int ex_k = 0; ex_k < sizeK; ex_k+=ACCESS_K){
			int curKsz=ACCESS_K;
			if((ex_k+ACCESS_K)>sizeK)
			{
				curKsz=sizeK-ex_k;
			}
			for(int ex_j = 0; ex_j < sizeN_in16; ex_j+=ACCESS_N){
#pragma HLS LOOP_FLATTEN off
				int curNsz=ACCESS_N;
				if((ex_j+ACCESS_N)>sizeN_in16)
				{
					curNsz=sizeN_in16-ex_j;
				}
			for(int k = 0; k < curKsz; k++){

				vB_rd:
					for (int j = 0; j < curNsz; j++) {
#pragma HLS pipeline
#pragma HLS LOOP_TRIPCOUNT min = 1 max = curNsz
						vB_local[k][j] = in2[(int(((ex_k+k) * sizeN)/16))+ex_j+j];
					}
			}


			if((ex_k/ACCESS_K)%2)
			{
				dataflow_region(A_Stream, vB_local, vC_local, in1, in3, out, C_Stream_in, C_Stream_out, startRow, endRow, curKsz, curNsz, ex_k, ex_j, sizeN, sizeK, in_alpha_const, in_beta);
			}
			else
			{
				dataflow_region(A_Stream, vB_local, vC_local, in1, out, in3, C_Stream_in, C_Stream_out, startRow, endRow, curKsz, curNsz, ex_k, ex_j, sizeN, sizeK, in_alpha_const, in_beta);
			}
		}
		}
    }
}
