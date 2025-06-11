#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include "simple_timer.hpp" 
#include <math.h>
// cublas headers
#include<cuda_runtime.h>
#include<cublas_v2.h>

/*
Author: Giacomo Zuccarino

Programmed as part of the assignment of the course P2.4_gpu2_24_25.

Professor: Piotr Luszczek.

Program: Master in High Performance Computing.

Institution: SISSA/ICTP, Trieste.
*/

/*
how to compile:
module load cuda/
nvcc -arch=sm_86 -o cuda_mat_mult.x matrix_multiply.cu -I. -lcublas

or

module load nvhpc/
nvcc -arch=sm_86 -o nvhpc_mat_mult.x matrix_multiply.cu -I. -lcublas
*/
/*
how to run:
./cuda_mat_mult.x 1000
*/

/*
    standard square matrix multiplication
*/
__global__ void matrixMultiply(float *A, float *B, float *C, std::size_t N) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    float value = 0.0f;
    if (row < N && col < N) {
        value = 0.0f;
        for (long int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

/*
    fill a matrix with random values
*/
void initializeMatrix(float *matrix, std::size_t N) {
    for (long int i = 0; i < N * N; ++i) {
        matrix[i] = rand() % 10;  // Fill with random values (0-9)
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Please provide the matrix size (N) as a command line argument." << std::endl;
        return 1;
    }

    // Get matrix size N from command line argument
    std::size_t N = std::stol(argv[1]);

    // Allocate matrices and store them linearly in vectors
    std::vector<float> matA(N * N,1.0);
    std::vector<float> matB(N * N,2.0);
    std::vector<float> matC(N * N,0.0);

    // Define block and grid size (16x16 threads per block)
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Time computation + memory copy
    {
        SimpleTimer t{"NAIVE -- computation + data transfer"};
        // Copy matrices from host to device
        cudaMemcpy(d_A, matA.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matB.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        // Perform multiplication
        matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

        // Copy the result back to host
        cudaMemcpy(matC.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize(); // Wait for GPU to finish;

    std::cout << "Result matrix C[0][0] = " << matC[0] << std::endl;

    /* CUBLAS SGEMM */

    // Create Handler
    cublasHandle_t cuda_handler;
    cublasCreate(&cuda_handler);
    float alpha = 1.0;
    float beta = 0.0;

    // Time computation + memory copy
    {
        SimpleTimer t{"CUBLAS -- computation + data transfer"};

        // Copy matrices from host to device
        cudaMemcpy(d_A, matA.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matB.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        // Perform multiplication
        cublasSgemm(cuda_handler,
                    CUBLAS_OP_T,
                    CUBLAS_OP_T,
                    N, // M --> number of Rows of A
                    N, // N --> number of Cols of B
                    N, // K --> number of Cols of A = number of Rows of B
                    &alpha,
                    d_A, // pointer to A memory storage location
                    N, // M -- > Leading dimension of A (number of rows in  A)
                    d_B, // pointer to B memory storage location
                    N, // K -- > Leading dimension of B (number of rows in  B)
                    &beta,
                    d_C, // pointer to C memory storage location
                    N); // M -- > Leading dimension of C (number of rows in  C)

        // Copy the result back to host
        cudaMemcpy(matC.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize(); // Wait for GPU to finish;

    std::cout << "Result matrix C[0][0] = " << matC[0] << std::endl;

    // Time computation + memory copy
    {
        SimpleTimer t{"CUBLAS (diagonal) -- computation + data transfer"};
        // Copy matrices from host to device
        cudaMemcpy(d_A, matA.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matB.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        // Perform multiplication
        cublasStrmm(cuda_handler,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT,
                    N, N,
                    &alpha,
                    d_A, N,
                    d_B, N,
                    d_C, N);

        // Copy the result back to host
        cudaMemcpy(matC.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize(); // Wait for GPU to finish;

    std::cout << "Result matrix C[0][0] = " << matC[0] << std::endl;

    size_t total_time = timing_table["NAIVE -- computation + data transfer"].time;
    std::cout << "giga flops/s (NAIVE): " << 2*N*N*N / (total_time *std::pow(10,-6)* std::pow(10,9)) << std::endl;

    size_t total_time_diag = timing_table["CUBLAS (diagonal) -- computation + data transfer"].time;
    std::cout << "giga flops/s (CUBLAS diagonal): " << N*N*(N+1) / (total_time_diag *std::pow(10,-6)* std::pow(10,9)) << std::endl; 

    size_t total_time_cublas = timing_table["CUBLAS -- computation + data transfer"].time;
    std::cout << "giga flops/s (CUBLAS): " << 2*N*N*N / (total_time_cublas *std::pow(10,-6)* std::pow(10,9)) << std::endl;

    // Finalize
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    SimpleTimer::print_timing_results();
    return 0;
}
