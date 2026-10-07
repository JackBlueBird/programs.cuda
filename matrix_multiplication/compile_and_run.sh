#!/bin/bash
# usage: ./compile_and_run.sh [N] [ITERATIONS] [WARMUP]   (default N=1000, ITERATIONS=10, WARMUP=5)
set -e

# on the cluster load the CUDA module; locally use /usr/local/cuda
if command -v module > /dev/null 2>&1; then
    module load cuda/12.9
else
    export PATH=/usr/local/cuda/bin:$PATH
fi

N=${1:-1000}
ITERATIONS=${2:-10}
WARMUP=${3:-}   # optional: if empty the program uses its default (5)

# sm_89 = Ada (RTX 4050 laptop); change for other GPUs (e.g. sm_86 on the cluster)
nvcc -std=c++17 -arch=sm_89 -o matrix_multiply_REDONE.x matrix_multiply_REDONE.cu -I. -lcublas
./matrix_multiply_REDONE.x "$N" "$ITERATIONS" $WARMUP
