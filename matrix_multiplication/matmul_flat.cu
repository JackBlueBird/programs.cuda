/*
    Minimal C = A * B on GPU with cublas: one multiplication, everything in main.
    Simplified version of matrix_multiply_REDONE.cu (no error checks, no warm up, no benchmark).

    Compile: nvcc -std=c++17 -arch=sm_89 matmul_flat.cu -lcublas -o matmul_flat.x
    Run:     ./matmul_flat.x [N]      (default N = 1000)
*/

#include <iostream>       // std::cout
#include <string>         // std::stoul to read N
#include <vector>         // std::vector for host matrices
#include <random>         // random values for A and B
#include <chrono>         // wall-clock timer
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, cudaFree
#include <cublas_v2.h>    // cublasCreate, cublasSgemm, cublasDestroy

int main(int argc, char* argv[]) {
    // matrix size as size_t (unsigned, 64 bit): sizes and indices computed from N never overflow.
    // note: no check on the input, a negative value would wrap around to a huge number
    size_t N = (argc > 1) ? std::stoul(argv[1]) : 1000;
    std::cout << "Running with N = " << N << std::endl;
    // compute number of bytes in a matrix of N x N floats
    size_t bytes = N * N * sizeof(float);

    // 1. host (CPU) matrices, row major: element (i, j) is at i * N + j
    std::vector<float> A(N * N), B(N * N), C(N * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : A) x = dist(gen);
    for (auto &x : B) x = dist(gen);

    // 2. device (GPU) matrices: these pointers are valid only on the GPU, never dereference them on the CPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // cublas context, needed by every cublas call
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    auto start = std::chrono::steady_clock::now();

    // 3. copy A and B host -> device
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    // 4. C = A * B on the GPU.
    //    cublas is column major: it reads our row major buffers as A^T and B^T.
    //    Passing B first computes B^T * A^T = (A * B)^T in column major,
    //    which in memory is exactly C = A * B in row major.
    //    cublas takes sizes and leading dimensions as int: explicit cast (fine for N < 2^31)
    const int n = static_cast<int>(N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, d_B, n, d_A, n, &beta, d_C, n);

    // 5. copy C device -> host; this waits for the (asynchronous) cublas call to finish
    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    auto stop = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    // 6. check one element on the CPU: C[0][0] = sum_k A[0][k] * B[k][0]
    double reference = 0.0;
    for (size_t k = 0; k < N; ++k) reference += double(A[k]) * B[k * N];
    std::cout << "C[0][0] GPU = " << C[0] << ", CPU = " << reference << std::endl;

    // time of one call, including copies and first-call initialization (no warm up)
    std::cout << "N = " << N << ", time = " << ms << " ms, "
              << 2.0 * N * N * N / (ms * 1e6) << " GFLOP/s" << std::endl;

    // 7. free device memory and cublas context
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
