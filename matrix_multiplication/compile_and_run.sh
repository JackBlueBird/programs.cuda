#!/bin/bash
# Compile and run both programs:
#   matmul_flat.x            minimal version, one multiplication
#   matrix_multiply_REDONE.x benchmark
# usage: ./compile_and_run.sh [N] [ITERATIONS] [WARMUP]   (default N=1000, ITERATIONS=10, WARMUP=5)
#   N is used by both programs, ITERATIONS and WARMUP only by the benchmark
set -e   # stop at the first error (e.g. a failed compilation)

# on the cluster load the CUDA module; locally use /usr/local/cuda
if command -v module > /dev/null 2>&1; then
    module load cuda/12.9
else
    export PATH=/usr/local/cuda/bin:$PATH
fi

N=${1:-1000}
ITERATIONS=${2:-10}
WARMUP=${3:-}   # optional: if empty the program uses its default (5)

# nvcc flags:
#   -std=c++17     C++ standard used by the code
#   -arch=sm_89    GPU architecture: sm_89 = Ada (RTX 4050 laptop); change for other GPUs (e.g. sm_86 on the cluster)
#   -I.            look for included headers (simple_timer.hpp) also in the current directory
#   -lcublas       link the cuBLAS library
#   -o name.x      name of the executable

echo "=== matmul_flat ==="
nvcc -std=c++17 -arch=sm_89 -o matmul_flat.x matmul_flat.cu -lcublas
./matmul_flat.x "$N"

echo "=== matrix_multiply_REDONE ==="
nvcc -std=c++17 -arch=sm_89 -o matrix_multiply_REDONE.x matrix_multiply_REDONE.cu -I. -lcublas
./matrix_multiply_REDONE.x "$N" "$ITERATIONS" $WARMUP
