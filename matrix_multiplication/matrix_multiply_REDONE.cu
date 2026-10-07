#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "simple_timer.hpp" 
#include <math.h>
// cublas headers
#include<cuda_runtime.h>
#include<cublas_v2.h>


/*
    test parameters from command line
*/
struct TestConfig {
    std::size_t N;      // size of the square matrices
    int n_iterations;   // number of timed iterations
    int n_warmup = 5;   // number of untimed warm up iterations, fixed
};

/*
    host matrices A, B and C (N x N, row major), c++ vectors
*/
struct HostMatrices {
    std::vector<float> A, B, C;
};

/*
    device matrices A, B and C (N x N, row major), c-style pointers, and cublas handler
*/
struct DeviceMatrices {
    float *A, *B, *C;
    cublasHandle_t cuda_handler;
};

/*
    C = A * B with cublas, A, B and C square N x N row major, already allocated on device
*/
void cublas_matmul(cublasHandle_t cuda_handler, const float *d_A, const float *d_B, float *d_C, std::size_t N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

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
}

/*
    H2D copy + C = A * B with cublas + D2H copy, device matrices already allocated
*/
void cublas_matmul_with_transfer(HostMatrices &host, DeviceMatrices &dev, std::size_t N) {
    // Copy matrices from host to device
    cudaMemcpy(dev.A, host.A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.B, host.B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform multiplication
    cublas_matmul(dev.cuda_handler, dev.A, dev.B, dev.C, N);

    // Copy the result back to host
    cudaMemcpy(host.C.data(), dev.C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
}

/*
    read command line arguments: matrix size N and number of timed iterations
    exits the program if arguments are missing or not valid
*/
TestConfig read_arguments(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix size N> <number of iterations>" << std::endl;
        std::exit(1);
    }
    long N;
    int n_iterations;
    try {
        N = std::stol(argv[1]);
        n_iterations = std::stoi(argv[2]);
    } catch (const std::exception &e) {
        std::cerr << "Arguments must be integer numbers." << std::endl;
        std::exit(1);
    }
    // check sign before converting N to std::size_t (a negative value would wrap around)
    if (N <= 0 || n_iterations <= 0) {
        std::cerr << "Matrix size and number of iterations must be positive." << std::endl;
        std::exit(1);
    }
    TestConfig config;
    config.N = static_cast<std::size_t>(N);
    config.n_iterations = n_iterations;
    return config;
}

/*
    allocate host matrices A, B and C (N x N, row major), set to zero
*/
HostMatrices setup_host(std::size_t N) {
    HostMatrices host;
    host.A.assign(N * N, 0.0f);
    host.B.assign(N * N, 0.0f);
    host.C.assign(N * N, 0.0f);
    return host;
}

/*
    allocate device matrices A, B and C (N x N) and create the cublas handler
*/
DeviceMatrices setup_device(std::size_t N) {
    DeviceMatrices dev;
    cudaMalloc((void**)&dev.A, N * N * sizeof(float));
    cudaMalloc((void**)&dev.B, N * N * sizeof(float));
    cudaMalloc((void**)&dev.C, N * N * sizeof(float));
    cublasCreate(&dev.cuda_handler);
    return dev;
}

/*
    destroy the cublas handler and free device matrices
*/
void cleanup_device(DeviceMatrices &dev) {
    cublasDestroy(dev.cuda_handler);
    cudaFree(dev.A);
    cudaFree(dev.B);
    cudaFree(dev.C);
}

/*
    print C[0][0], average giga flops/s for the given timer label and the timing table
*/
void print_results(std::size_t N, const HostMatrices &host, const std::string &timer_label) {
    std::cout << "Result matrix C[0][0] = " << host.C[0] << std::endl;
    // average time per call in microseconds (double, no truncation)
    double avg_time = SimpleTimer::average_us(timer_label);
    // flops / (time_us * 1e-6) / 1e9 = flops / (time_us * 1e3)
    std::cout << "giga flops/s (CUBLAS): " << 2.0*N*N*N / (avg_time * 1e3) << std::endl;
    SimpleTimer::print_timing_results();
}

int main(int argc, char* argv[]) {
    TestConfig config = read_arguments(argc, argv);
    HostMatrices host = setup_host(config.N);
    DeviceMatrices dev = setup_device(config.N);

    // Warm up: first calls pay cublas initialization and first transfers, not timed
    for (int i = 0; i < config.n_warmup; ++i) {
        cublas_matmul_with_transfer(host, dev, config.N);
    }

    // Time computation + memory copy
    for (int i = 0; i < config.n_iterations; ++i) {
        SimpleTimer t{"CUBLAS -- computation + data transfer"};
        cublas_matmul_with_transfer(host, dev, config.N);
    }

    print_results(config.N, host, "CUBLAS -- computation + data transfer");
    cleanup_device(dev);
    return 0;
}
