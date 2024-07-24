#include <stdio.h>
#include <memory>
#include <string>
#include <iostream>
#include <hls_stream.h>
// Including to use ap_uint<> datatype
#include <ap_int.h>
#include <math.h>

// Define data width and associated types
#define DATAWIDTH 512
typedef ap_uint<DATAWIDTH> uint512_dt;

#define ACCESS_K 32
#define ACCESS_N 64

#define UNROLL_N 2
#define UNROLL_K 4

#define BUFSIZE (1024 * 1024 * 32)

#define DATATYPE_SIZE 32
#define VECTOR_SIZE (DATAWIDTH / DATATYPE_SIZE) // vector size is 16 (512/32 = 16)

#define BUFSIZE_A (1024 * 128)
#define BUFSIZE_B (1024 * 128)
#define BUFSIZE_C (1024 * 128)
#define BUFSIZE_D (1024 * 128)

typedef ap_uint<DATATYPE_SIZE> din_type;

// Function prototypes for conversion between uint and float
float uint_to_float(din_type x);
unsigned int float_to_uint(float x);

extern "C"
{
// FPGA kernel for vector addition
void wide_vadd(
    const uint512_dt *in1, // Read-Only Vector 1 (Matrix A)
    const uint512_dt *in2, // Read-Only Vector 2 (Matrix B)
    uint512_dt *in3, // Read-Only Vector 3 (Matrix C)
    uint512_dt *out, // Output Result (Matrix D)
    int sizeK, // Number of columns in matrix A and number of rows in matrix B
    int sizeN, // Number of columns in matrix B and matrix C
    int startRow, // Starting row for processing
    int endRow // Ending row for processing
    );
}

// Software reference implementation for matrix multiplication
void test_kernel_big(int M, int N, int K, float *A, float *B, float *C, float *D)
{
    bool verified = true;

    // Compute C = A * B
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[m * N + j] += A[m * K + k] * B[k * N + j];
            }
        }
    }

    int count = 0;
    // Verify the result
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < N; ++j) {
            if (abs(D[m * N + j] - C[m * N + j]) > 1e-3) {
                verified = false;
                std::cout << "ERROR: software and hardware vadd do not match: "
                          << D[m * N + j] << " != " << C[m * N + j] << " at position " << m * N + j << std::endl;
                std::cout << std::endl
                          << "OCL-mapped contiguous buffer example complete! (with errors)"
                          << std::endl
                          << std::endl;
                count++;
                if (count == 4) {
                    goto test;
                }
            }
        }
    }

test:
    if (verified) {
        std::cout << std::endl
                  << "OCL-mapped contiguous buffer example complete!"
                  << std::endl
                  << std::endl;
    } else {
        std::cout << std::endl
                  << "OCL-mapped contiguous buffer example complete! (with errors)"
                  << std::endl
                  << std::endl;
    }
}

int main()
{
    int krnl_M = 128; // Number of rows in matrix A and matrix D
    int krnl_K = 128; // Number of columns in matrix A and number of rows in matrix B
    int krnl_N = 256; // Number of columns in matrix B and matrix C

    // Allocate memory for the matrices
    float *a = (float *)malloc(BUFSIZE_A * sizeof(float));
    float *b = (float *)malloc(BUFSIZE_B * sizeof(float));
    float *c = (float *)malloc(BUFSIZE_C * sizeof(float));
    float *d = (float *)malloc(BUFSIZE_D * sizeof(float));
    float *e = (float *)malloc(BUFSIZE_D * sizeof(float));

    // Allocate memory for the 512-bit packed data
    uint512_dt *a_512 = (uint512_dt *)malloc(BUFSIZE_A / VECTOR_SIZE * sizeof(uint512_dt));
    uint512_dt *b_512 = (uint512_dt *)malloc(BUFSIZE_B / VECTOR_SIZE * sizeof(uint512_dt));
    uint512_dt *c_512 = (uint512_dt *)malloc(BUFSIZE_C / VECTOR_SIZE * sizeof(uint512_dt));
    uint512_dt *d_512 = (uint512_dt *)malloc(BUFSIZE_D / VECTOR_SIZE * sizeof(uint512_dt));

    // Initialize matrix A
    for (int i = 0; i < BUFSIZE_A; i++) {
        if (i % 3 == 0) {
            a[i] = 0.1;
        } else if (i % 3 == 1) {
            a[i] = 0.2;
        } else {
            a[i] = 0.3;
        }
    }

    // Pack matrix A into 512-bit data
    for (int i = 0; i < BUFSIZE_A / VECTOR_SIZE; i++) {
        uint512_dt tmp_a;
        for (int vector = 0; vector < VECTOR_SIZE; vector++) {
            if ((i * VECTOR_SIZE + vector) % 3 == 0) {
                tmp_a.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(0.1);
            } else if ((i * VECTOR_SIZE + vector) % 3 == 1) {
                tmp_a.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(0.2);
            } else {
                tmp_a.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(0.3);
            }
        }
        a_512[i] = tmp_a;
    }

    // Initialize matrix B
    for (int i = 0; i < BUFSIZE_B; i++) {
        if (i % 3 == 0) {
            b[i] = 1;
        } else if (i % 3 == 1) {
            b[i] = 0.5;
        } else {
            b[i] = 0.5;
        }
    }

    // Pack matrix B into 512-bit data
    for (int i = 0; i < BUFSIZE_B / VECTOR_SIZE; i++) {
        uint512_dt tmp_b;
        for (int vector = 0; vector < VECTOR_SIZE; vector++) {
            if ((i * VECTOR_SIZE + vector) % 3 == 0) {
                tmp_b.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(1);
            } else if ((i * VECTOR_SIZE + vector) % 3 == 1) {
                tmp_b.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(0.5);
            } else {
                tmp_b.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(0.5);
            }
        }
        b_512[i] = tmp_b;
    }

    // Initialize matrix C (reference)
    for (int i = 0; i < BUFSIZE_C; i++) {
        if (i % 3 == 0) {
            e[i] = 0.2;
        } else if (i % 3 == 1) {
            e[i] = 0.1;
        } else {
            e[i] = 1;
        }
    }

    // Pack matrix C into 512-bit data
    for (int i = 0; i < BUFSIZE_D / VECTOR_SIZE; i++) {
        uint512_dt tmp_d;
        for (int vector = 0; vector < VECTOR_SIZE; vector++) {
            if ((i * VECTOR_SIZE + vector) % 3 == 0) {
                tmp_d.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(0.2); // 0.2
            } else if ((i * VECTOR_SIZE + vector) % 3 == 1) {
                tmp_d.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(0.1); // 0.1
            } else {
                tmp_d.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(1); // 1
            }
        }
        c_512[i] = tmp_d;
        d_512[i] = tmp_d;
    }

    // Call the FPGA kernel
    wide_vadd(a_512, b_512, c_512, d_512, krnl_K, krnl_N, 0, krnl_M);

    // Convert D array from 512-bit to 32-bit
    for (int i = 0; i < BUFSIZE_D / VECTOR_SIZE; i++) {
        uint512_dt tmp_d = d_512[i];
        for (int vector = 0; vector < VECTOR_SIZE; vector++) {
            d[i * VECTOR_SIZE + vector] = uint_to_float(tmp_d.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE));
        }
    }

    // Convert C array from 512-bit to 32-bit
    for (int i = 0; i < BUFSIZE_D / VECTOR_SIZE; i++) {
        uint512_dt tmp_c = c_512[i];
        for (int vector = 0; vector < VECTOR_SIZE; vector++) {
            c[i * VECTOR_SIZE + vector] = uint_to_float(tmp_c.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE));
        }
    }

    // Perform verification with the software implementation
    if ((int(ceil(krnl_K / ACCESS_K))) % 2) {
        test_kernel_big(krnl_M, krnl_N, krnl_K, a, b, e, c);
    } else {
        test_kernel_big(krnl_M, krnl_N, krnl_K, a, b, e, d);
    }

    return 0;
}
