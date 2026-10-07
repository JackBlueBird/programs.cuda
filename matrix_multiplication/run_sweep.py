#!/usr/bin/env python3
"""
Run matrix_multiply_REDONE.x for several matrix sizes N and save the results in a CSV file.

The executable must already be compiled (e.g. with ./compile_and_run.sh).
If the `module` command exists (cluster), the CUDA module is loaded in the same shell
that runs the executable; locally (no modules) the executable is run directly.

usage:
    python3 run_sweep.py                                  # N = 500, 1000, ..., 5000
    python3 run_sweep.py --sizes 1000 2000 --iterations 20 --warmup 5 --output results.csv
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

EXECUTABLE = Path(__file__).resolve().parent / "matrix_multiply_REDONE.x"
MODULE = "cuda/12.9"
MAX_N = 10000  # safety limit: 3 float matrices of N x N = 12 * N^2 bytes on host and device

# lines printed by the program, e.g.
#   giga flops/s (CUBLAS -- computation + data transfer): 834.376
#   CUBLAS -- computation + data transfer -> Total time: 11985 μs, Average time: 2397 μs, Calls: 5
GFLOPS_RE = re.compile(r"giga flops/s \(.*\):\s*([0-9.eE+-]+)")
AVG_TIME_RE = re.compile(r"Average time:\s*([0-9.eE+-]+)")


def build_command(N, iterations, warmup):
    """shell command for one run, with module load if the module system is available"""
    run = f"{EXECUTABLE} {N} {iterations}"
    if warmup is not None:
        run += f" {warmup}"
    # `module` is a shell function defined by login scripts: check it inside a login shell
    return f"if command -v module > /dev/null 2>&1; then module load {MODULE}; fi; {run}"


def run_one(N, iterations, warmup):
    """run the executable once, return (average time in us, GFLOP/s)"""
    result = subprocess.run(["bash", "-lc", build_command(N, iterations, warmup)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Run with N={N} failed (exit code {result.returncode}):\n{result.stdout}{result.stderr}")

    gflops = GFLOPS_RE.search(result.stdout)
    avg_time = AVG_TIME_RE.search(result.stdout)
    if gflops is None or avg_time is None:
        sys.exit(f"Could not parse the output for N={N}:\n{result.stdout}")
    return float(avg_time.group(1)), float(gflops.group(1))


def main():
    parser = argparse.ArgumentParser(description="Sweep of matrix_multiply_REDONE.x over matrix sizes")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(range(500, 5001, 500)),
                        help="matrix sizes N (default: 500 1000 ... 5000)")
    parser.add_argument("--iterations", type=int, default=1000, help="timed iterations per run (default: 1000)")
    parser.add_argument("--warmup", type=int, default=100, help="warm up iterations (default: 100)")
    parser.add_argument("--output", default="sweep_results.csv", help="output CSV file (default: sweep_results.csv)")
    args = parser.parse_args()

    if not EXECUTABLE.exists():
        sys.exit(f"{EXECUTABLE} not found: compile it first (e.g. ./compile_and_run.sh)")
    too_big = [N for N in args.sizes if N > MAX_N]
    if too_big:
        sys.exit(f"Sizes above the safety limit {MAX_N}: {too_big}")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "iterations", "avg_time_us", "gflops"])
        for N in args.sizes:
            avg_time, gflops = run_one(N, args.iterations, args.warmup)
            writer.writerow([N, args.iterations, avg_time, gflops])
            print(f"N = {N:6d}   avg time = {avg_time:12.1f} us   {gflops:10.2f} GFLOP/s")

    print(f"Results saved in {args.output}")


if __name__ == "__main__":
    main()
