#!/usr/bin/env python3
"""
Run matrix_multiply_REDONE.x for several matrix sizes N and save the results in a CSV file.

The executable must already be compiled (e.g. with ./compile_and_run.sh).
If the `module` command exists (cluster), the CUDA module is loaded in the same shell
that runs the executable; locally (no modules) the executable is run directly.

While each run executes, the GPU is sampled with NVML (package nvidia-ml-py, imported as pynvml):
SM clock, temperature, power and throttle reasons are summarized per N in the CSV.
If pynvml is not installed (or --no-monitor is given) those columns are left empty.
Note: samples cover the whole run of the executable (start-up, warm up, timed iterations, verification).

usage:
    python3 run_sweep.py                                  # N = 500, 1000, ..., 15000
    python3 run_sweep.py --sizes 1000 2000 --iterations 20 --warmup 5 --output results.csv
"""

import argparse
import csv
import re
import subprocess
import sys
import threading
from pathlib import Path

try:
    import pynvml
except ImportError:
    pynvml = None

EXECUTABLE = Path(__file__).resolve().parent / "matrix_multiply_REDONE.x"
MODULE = "cuda/12.9"
MAX_N = 15000  # safety limit: 3 float matrices of N x N = 12 * N^2 bytes on host and device
GPU_INDEX = 0  # GPU sampled by the monitor
SAMPLE_INTERVAL_S = 0.1

# lines printed by the program, e.g.
#   giga flops/s (CUBLAS -- computation + data transfer): 834.376
#   CUBLAS -- computation + data transfer -> Total time: 11985 μs, Average time: 2397 μs, Calls: 5
GFLOPS_RE = re.compile(r"giga flops/s \(.*\):\s*([0-9.eE+-]+)")
AVG_TIME_RE = re.compile(r"Average time:\s*([0-9.eE+-]+)")

# NVML throttle reason bits (nvml.h, nvmlClocksThrottleReason* / nvmlClocksEventReason*)
POWER_REASONS = 0x4 | 0x80     # SwPowerCap | HwPowerBrakeSlowdown
THERMAL_REASONS = 0x20 | 0x40  # SwThermalSlowdown | HwThermalSlowdown
HW_SLOWDOWN = 0x8              # HwSlowdown (thermal or power, reported by hardware)

CSV_HEADER = ["N", "iterations", "avg_time_us", "gflops",
              "sm_clock_avg_mhz", "sm_clock_min_mhz", "temp_max_c", "power_avg_w", "throttle"]


class GpuMonitor:
    """samples the GPU in a background thread between start() and stop()"""

    def __init__(self, handle):
        self.handle = handle
        self.samples = []  # (sm clock MHz, temperature C, power W, throttle reasons bitmask)
        self._stop = threading.Event()
        self._thread = None

    def _read_reasons(self):
        # the function was renamed in recent NVML versions: use whichever exists
        for name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            if hasattr(pynvml, name):
                return getattr(pynvml, name)(self.handle)
        return 0

    def _loop(self):
        while not self._stop.is_set():
            clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W
            self.samples.append((clock, temp, power, self._read_reasons()))
            self._stop.wait(SAMPLE_INTERVAL_S)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """stop sampling and return the summary columns for the CSV"""
        self._stop.set()
        self._thread.join()
        if not self.samples:
            return ["", "", "", "", ""]
        clocks = [s[0] for s in self.samples]
        reasons = 0
        for s in self.samples:
            reasons |= s[3]
        throttle = []
        if reasons & POWER_REASONS:
            throttle.append("power")
        if reasons & THERMAL_REASONS:
            throttle.append("thermal")
        if reasons & HW_SLOWDOWN:
            throttle.append("hw_slowdown")
        return [round(sum(clocks) / len(clocks)),
                min(clocks),
                max(s[1] for s in self.samples),
                round(sum(s[2] for s in self.samples) / len(self.samples), 1),
                "+".join(throttle) if throttle else "none"]


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


def create_monitor(enabled):
    """GpuMonitor on GPU_INDEX, or None if disabled or NVML is not available"""
    if not enabled:
        return None
    if pynvml is None:
        print("pynvml not installed (pip install nvidia-ml-py): GPU monitoring disabled")
        return None
    pynvml.nvmlInit()
    return GpuMonitor(pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX))


def main():
    parser = argparse.ArgumentParser(description="Sweep of matrix_multiply_REDONE.x over matrix sizes")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(range(500, 15001, 500)),
                        help="matrix sizes N (default: 500 1000 ... 10000)")
    parser.add_argument("--iterations", type=int, default=10, help="timed iterations per run (default: 10)")
    parser.add_argument("--warmup", type=int, default=10, help="warm up iterations (default: 10)")
    parser.add_argument("--output", default="sweep_results.csv", help="output CSV file (default: sweep_results.csv)")
    parser.add_argument("--no-monitor", action="store_true", help="do not sample the GPU with NVML")
    args = parser.parse_args()

    if not EXECUTABLE.exists():
        sys.exit(f"{EXECUTABLE} not found: compile it first (e.g. ./compile_and_run.sh)")
    too_big = [N for N in args.sizes if N > MAX_N]
    if too_big:
        sys.exit(f"Sizes above the safety limit {MAX_N}: {too_big}")

    monitor = create_monitor(not args.no_monitor)

    try:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for N in args.sizes:
                if monitor:
                    monitor.start()
                avg_time, gflops = run_one(N, args.iterations, args.warmup)
                gpu = monitor.stop() if monitor else ["", "", "", "", ""]
                writer.writerow([N, args.iterations, avg_time, gflops] + gpu)
                line = f"N = {N:6d}   avg time = {avg_time:12.1f} us   {gflops:10.2f} GFLOP/s"
                if monitor:
                    line += f"   SM clock avg {gpu[0]} MHz (min {gpu[1]})   {gpu[2]} C   {gpu[3]} W   throttle: {gpu[4]}"
                print(line)
    finally:
        if monitor:
            pynvml.nvmlShutdown()

    print(f"Results saved in {args.output}")


if __name__ == "__main__":
    main()
