# Matrix multiplication benchmark with cuBLAS

`C = A * B` for square `N x N` float matrices on one NVIDIA GPU (cuBLAS SGEMM), timed including host <-> device copies.

- `matmul_flat.cu`: minimal version, one multiplication in a single `main()` (start here if new to CUDA)
- `matrix_multiply_REDONE.cu`: the benchmark (start reading above `main()`)
- `simple_timer.hpp`: timer used by the benchmark
- `compile_and_run.sh`: compile and run both programs once
- `run_sweep.py`: run for many N, write a CSV
- `plot_sweep.py`: plots from the CSV
- `compare_sweeps.py`: compares two CSV files in one multi-panel figure

## Executable

- **Driver**: NVIDIA GPU with driver installed
    - check: `nvidia-smi`
- **CUDA Toolkit**: provides both `nvcc` and cuBLAS
    - local: install from <https://developer.nvidia.com/cuda-downloads>, then `export PATH=/usr/local/cuda/bin:$PATH`
    - cluster: `module load cuda/12.9`
    - check: `nvcc --version`
- **Compile and run**
    - compile: `nvcc -std=c++17 -arch=sm_89 -I. matrix_multiply_REDONE.cu -lcublas -o matrix_multiply_REDONE.x`
    - run: `./matrix_multiply_REDONE.x 2000 10 5` (N = 2000, 10 timed iterations, 5 warm up; warm up is optional, default 5)
    - both in one step: `./compile_and_run.sh [N] [ITERATIONS] [WARMUP]` (defaults 1000, 10, 5)
    - `-arch=sm_89` in the script = compute capability 8.9 (RTX 40); change it for other GPUs
- **Memory**: `12 * N^2` bytes on host and on device (N = 10000 -> 1.2 GB)

## Python scripts

- **Packages**: `pip install matplotlib nvidia-ml-py`
    - `nvidia-ml-py` is optional: without it the GPU monitor columns stay empty
- **Usage** (compile first)
    - `python3 run_sweep.py --sizes 1000 2000 4000 --iterations 10 --warmup 5 --output results.csv`
    - `python3 plot_sweep.py --input results.csv --prefix results`
    - `python3 compare_sweeps.py powersaver.csv performance.csv --labels "power saver" "performance"`
- **Laptop**: the power mode (power saver / performance) changes the results, note it with each CSV

## Acronyms

  - **BLAS**: Basic Linear Algebra Subprograms, the standard interface for vector and matrix operations (column major)
  - **cuBLAS**: NVIDIA implementation of BLAS on GPU
  - **CSV**: Comma-Separated Values, the text table format of the results
  - **CUDA**: NVIDIA platform for programming GPUs
  - **D2H / H2D**: data copy device -> host / host -> device
  - **FLOP**: floating point operation; a matrix multiplication of size N does `2 * N^3`
  - **FP32**: 32 bit floating point, `float` (single precision)
  - **GFLOP/s**: 10^9 floating point operations per second
  - **host / device**: CPU with its RAM / GPU with its VRAM
  - **nvcc**: NVIDIA CUDA compiler
  - **NVML**: NVIDIA Management Library, reads GPU clock, temperature and power
  - **OOM**: Out Of Memory; the system kills the process (exit code 137)
  - **PCIe**: PCI Express, the bus between CPU and GPU used by the copies
  - **SGEMM**: Single precision GEneral Matrix-Matrix multiply, `C = alpha * A * B + beta * C`
  - **SM**: Streaming Multiprocessor, the GPU compute unit; the SM clock is the speed of its cores
  - **sm_XY**: GPU compute capability X.Y, selected with `nvcc -arch=sm_XY`
  - **TGP**: Total Graphics Power, the power limit of a laptop GPU
  - **throttling**: automatic clock reduction to stay within power or temperature limits
  - **VRAM**: GPU memory
