#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "simple_timer.hpp" 
#include <math.h>
// cublas headers
#include<cuda_runtime.h>
#include<cublas_v2.h>


int main (int argc, char* argv[]) {
    std::cout << "Hello World!" << std::endl;
    if (argc < 2) {
        std::cerr << "Please provide the matrix size (N) as a command line argument." << std::endl;
        return 1;
    }

    // Get matrix size N from command line argument

    std::size_t N = 100; // Size of the square matrices
    N = std::stol(argv[1]);
    
    // host storage for matrices, c++ vectors
    std::vector<float> mat_A(N * N, 0.0f);
    std::vector<float> mat_B(N * N, 0.0f);
    std::vector<float> mat_C(N * N, 0.0f);

    // device storage for matrices, c-style pointers
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cublasHandle_t cuda_handler;
    cublasCreate(&cuda_handler);
    float alpha = 1.0;
    float beta = 0.0;

    // Time computation + memory copy
    for (int i = 0; i < 10; ++i) {
        {
            SimpleTimer t{"CUBLAS -- computation + data transfer"};

            // Copy matrices from host to device
            cudaMemcpy(d_A, mat_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, mat_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

            // Perform multiplication alpha * B^T * A^T + beta * C^T = C^T where A, B and C are row major stored
            // cublas is col major: it reads our row major buffers as A^T, B^T and writes C^T col major,
            // which in memory is exactly C row major
            cublasSgemm(cuda_handler,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        N,      // M --> number of Rows of C^T (= Cols of C)
                        N,      // N --> number of Cols of C^T (= Rows of C)
                        N,      // K --> common dimension (Cols of A = Rows of B)
                        &alpha,
                        d_B,    // pointer to B memory storage location (seen by cublas as B^T)
                        N,      // N --> Leading dimension of B (number of cols of B, row major)
                        d_A,    // pointer to A memory storage location (seen by cublas as A^T)
                        N,      // K --> Leading dimension of A (number of cols of A, row major)
                        &beta,
                        d_C,    // pointer to C memory storage location (written by cublas as C^T)
                        N);     // N --> Leading dimension of C (number of cols of C, row major)

            // Copy the result back to host
            cudaMemcpy(mat_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    
    std::cout << "Result matrix C[0][0] = " << mat_C[0] << std::endl;
    // average time per call in microseconds (double, no truncation)
    double avg_time = SimpleTimer::average_us("CUBLAS -- computation + data transfer");
    // flops / (time_us * 1e-6) / 1e9 = flops / (time_us * 1e3)
    std::cout << "giga flops/s (CUBLAS): " << 2.0*N*N*N / (avg_time * 1e3) << std::endl;
    SimpleTimer::print_timing_results();
    // free memory on device
    cublasDestroy(cuda_handler);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}