// Benchmark of C = A * B on a single GPU with cublas: see the description above main()

#include <iostream>         // std::cout, std::cerr for output and error messages
#include <cstdlib>          // std::exit on invalid arguments or CUDA/cublas errors
#include <string>           // std::string, std::stol, std::stoi (argument parsing, timer label)
#include <vector>           // std::vector for host matrices
#include "simple_timer.hpp" // SimpleTimer: wall-clock timing and average per label
#include <cuda_runtime.h>   // cudaMalloc, cudaMemcpy, cudaFree, cudaError_t
#include <cublas_v2.h>      // cublasCreate, cublasSgemm, cublasDestroy, cublasHandle_t

/*
    error checking: on failure print file, line and error, then exit
*/
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << ": " << cudaGetErrorString(err) << std::endl;          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cublasGetStatusString(status) << std::endl;    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

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
    device matrices A, B and C (N x N, row major), c-style pointers, and cublas handle
*/
struct DeviceMatrices {
    float *A = nullptr, *B = nullptr, *C = nullptr;
    cublasHandle_t cublas_handle = nullptr;
};

/*
    C = A * B with cublas, A, B and C square N x N row major, already allocated on device
*/
void cublas_matmul(cublasHandle_t cublas_handle, const float *d_A, const float *d_B, float *d_C, std::size_t N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // A row major buffer read as col major is the transpose of the matrix.
    // cublas is col major: it reads our row major buffers as A^T, B^T and writes C^T col major,
    // which in memory is exactly C row major. So we ask for C^T = B^T * A^T (B first, no transposes):
    // alpha * B^T * A^T + beta * C^T = C^T
    // In the comments below m, n, k, lda, ldb, ldc are the cublas parameter names, N is our matrix size.
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,      // m --> number of Rows of C^T (= Cols of C)
                N,      // n --> number of Cols of C^T (= Rows of C)
                N,      // k --> common dimension (Cols of A = Rows of B)
                &alpha,
                d_B,    // pointer to B memory storage location (seen by cublas as B^T)
                N,      // ldb --> Leading dimension of B (number of cols of B, row major)
                d_A,    // pointer to A memory storage location (seen by cublas as A^T)
                N,      // lda --> Leading dimension of A (number of cols of A, row major)
                &beta,
                d_C,    // pointer to C memory storage location (written by cublas as C^T)
                N));    // ldc --> Leading dimension of C (number of cols of C, row major)
}

/*
    copy host -> device (H2D) + C = A * B with cublas + copy device -> host (D2H), device matrices already allocated
*/
void cublas_matmul_with_transfer(HostMatrices &host, DeviceMatrices &dev, std::size_t N) {
    // Copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(dev.A, host.A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.B, host.B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Perform multiplication
    cublas_matmul(dev.cublas_handle, dev.A, dev.B, dev.C, N);

    // Copy the result back to host.
    // cublas calls are asynchronous (return before the GPU finishes): this cudaMemcpy waits
    // for the multiplication to end, so when this function returns all GPU work is done
    CUDA_CHECK(cudaMemcpy(host.C.data(), dev.C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
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
    allocate device matrices A, B and C (N x N) and create the cublas handle
*/
DeviceMatrices setup_device(std::size_t N) {
    DeviceMatrices dev;
    CUDA_CHECK(cudaMalloc((void**)&dev.A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dev.B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dev.C, N * N * sizeof(float)));
    CUBLAS_CHECK(cublasCreate(&dev.cublas_handle));
    return dev;
}

/*
    destroy the cublas handle and free device matrices
*/
void cleanup_device(DeviceMatrices &dev) {
    CUBLAS_CHECK(cublasDestroy(dev.cublas_handle));
    CUDA_CHECK(cudaFree(dev.A));
    CUDA_CHECK(cudaFree(dev.B));
    CUDA_CHECK(cudaFree(dev.C));
}

/*
    print C[0][0], average giga flops/s for the given timer label and the timing table
*/
void print_results(std::size_t N, const HostMatrices &host, const std::string &timer_label) {
    std::cout << "Result matrix C[0][0] = " << host.C[0] << std::endl;
    // average time per call in microseconds (double, no truncation)
    double avg_time = SimpleTimer::average_us(timer_label);
    // flops / (time_us * 1e-6) / 1e9 = flops / (time_us * 1e3)
    std::cout << "giga flops/s (" << timer_label << "): " << 2.0*N*N*N / (avg_time * 1e3) << std::endl;
    SimpleTimer::print_timing_results();
}

/*
    Start reading here, then main, then the helper functions above.

    This program runs a benchmark for C = A * B executed on a single GPU:
    square N x N float matrices, multiplied with cublas, timed over several iterations.

    Flow:
    - allocate matrices on host (CPU)
    - copy A and B to device (GPU)
    - multiply C = A * B on device
    - copy C back to host

    Main terms:
    host            CPU and its RAM
    device          GPU and its memory (VRAM)
    H2D / D2H       copy host -> device / device -> host (cudaMemcpy)
    cudaMalloc      allocates memory on device; the pointer is valid only on device (do not dereference it on host)
    cudaFree        frees device memory
    cublas          NVIDIA linear algebra library (BLAS on GPU)
    cublas handle   cublas context, created once (cublasCreate) and passed to every cublas call
    Sgemm           single precision (float) general matrix-matrix multiply: C = alpha * A * B + beta * C
    row major       matrix stored row after row (C/C++); col major: column after column (Fortran, BLAS, cublas)
    leading dim.    memory distance between the start of two consecutive rows (row major) or columns (col major)
    asynchronous    cublas calls return before the GPU has finished; a later sync point (e.g. cudaMemcpy) waits
    warm up         untimed calls before measuring, to exclude one-time initialization costs
    GFLOP/s         10^9 floating point operations per second; matmul does 2*N^3 (N^3 multiplications + N^3 additions)

    Compile: nvcc -std=c++17 -arch=sm_89 -I. matrix_multiply_REDONE.cu -lcublas -o matrix_multiply_REDONE.x
    Run:     ./matrix_multiply_REDONE.x <N> <iterations>
*/
int main(int argc, char* argv[]) {
    TestConfig config = read_arguments(argc, argv);
    HostMatrices host = setup_host(config.N);
    DeviceMatrices dev = setup_device(config.N);

    // Warm up: first calls pay cublas initialization and first transfers, not timed
    for (int i = 0; i < config.n_warmup; ++i) {
        cublas_matmul_with_transfer(host, dev, config.N);
    }

    // Time computation + memory copy
    const std::string timer_label = "CUBLAS -- computation + data transfer";
    for (int i = 0; i < config.n_iterations; ++i) {
        SimpleTimer t{timer_label};
        cublas_matmul_with_transfer(host, dev, config.N);
    }

    print_results(config.N, host, timer_label);
    cleanup_device(dev);
    return 0;
}
