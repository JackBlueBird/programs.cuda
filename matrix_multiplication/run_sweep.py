#!/usr/bin/env python3
"""
Run matrix_multiply_REDONE.x for several matrix sizes N and save the results in a CSV file.

The executable must already be compiled (e.g. with ./compile_and_run.sh).
If the `module` command exists (cluster), the CUDA module is loaded in the same shell
that runs the executable; locally (no modules) the executable is run directly.

While each run executes, the GPU is sampled with NVML (package nvidia-ml-py, imported as pynvml)
every SAMPLE_INTERVAL_S seconds. Per N the CSV gets:
- SM clock average / minimum and power average over the ACTIVE samples only
  (GPU utilization >= ACTIVE_UTIL_PCT), so idle phases (program start-up, matrix generation on CPU,
  verification) do not dilute them; empty if no sample was active (very short runs)
- maximum temperature over all samples
- enforced power limit (the power cap applied by driver / power mode)
- percentage of active samples with each throttle reason (power, thermal, hardware slowdown)
If pynvml is not installed (or --no-monitor is given) those columns are left empty.

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
ACTIVE_UTIL_PCT = 50  # a sample counts as "GPU active" at or above this utilization

# timer labels printed by the program, one per measurement
LABEL_TRANSFER = "CUBLAS -- computation + data transfer"
LABEL_COMPUTE = "CUBLAS -- computation only"


def gflops_re(label):
    """matches e.g. 'giga flops/s (CUBLAS -- computation only): 834.376'"""
    return re.compile(r"giga flops/s \(" + re.escape(label) + r"\):\s*([0-9.eE+-]+)")


def avg_time_re(label):
    """matches e.g. 'CUBLAS -- computation only -> Total time: 11985 μs, Average time: 2397 μs, Calls: 5'"""
    return re.compile(re.escape(label) + r" -> Total time:.*?Average time:\s*([0-9.eE+-]+)")

# NVML throttle reason bits (nvml.h, nvmlClocksThrottleReason* / nvmlClocksEventReason*)
POWER_REASONS = 0x4 | 0x80     # SwPowerCap | HwPowerBrakeSlowdown
THERMAL_REASONS = 0x20 | 0x40  # SwThermalSlowdown | HwThermalSlowdown
HW_SLOWDOWN = 0x8              # HwSlowdown (thermal or power, reported by hardware)

MONITOR_COLUMNS = ["active_samples", "sm_clock_avg_mhz", "sm_clock_min_mhz", "temp_max_c",
                   "power_avg_w", "power_limit_w",
                   "throttle_power_pct", "throttle_thermal_pct", "throttle_hw_pct"]
# avg_time_us / gflops: computation + data transfer; compute_*: computation only
TIMING_COLUMNS = ["avg_time_us", "gflops", "compute_time_us", "compute_gflops"]
CSV_HEADER = ["N", "iterations"] + TIMING_COLUMNS + MONITOR_COLUMNS
EMPTY_MONITOR = {name: "" for name in MONITOR_COLUMNS}


class GpuMonitor:
    """samples the GPU in a background thread between start() and stop()"""

    def __init__(self, handle):
        self.handle = handle
        # (sm clock MHz, temperature C, power W, utilization %, throttle reasons bitmask)
        self.samples = []
        self.power_limit_w = None
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
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu  # % of time a kernel was running
            self.samples.append((clock, temp, power, util, self._read_reasons()))
            self._stop.wait(SAMPLE_INTERVAL_S)

    def start(self):
        self.samples = []
        # power cap applied now (depends on driver and laptop power mode)
        self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0  # mW -> W
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """stop sampling and return the summary columns for the CSV, as a dict"""
        self._stop.set()
        self._thread.join()
        summary = dict(EMPTY_MONITOR)
        if not self.samples:
            return summary
        summary["temp_max_c"] = max(s[1] for s in self.samples)
        summary["power_limit_w"] = round(self.power_limit_w, 1)

        active = [s for s in self.samples if s[3] >= ACTIVE_UTIL_PCT]
        summary["active_samples"] = len(active)
        if not active:
            return summary  # run too short to catch the GPU working: clock/power/throttle left empty

        def pct(mask):
            """percentage of active samples with any of the reason bits in mask"""
            return round(100.0 * sum(1 for s in active if s[4] & mask) / len(active))

        clocks = [s[0] for s in active]
        summary["sm_clock_avg_mhz"] = round(sum(clocks) / len(clocks))
        summary["sm_clock_min_mhz"] = min(clocks)
        summary["power_avg_w"] = round(sum(s[2] for s in active) / len(active), 1)
        summary["throttle_power_pct"] = pct(POWER_REASONS)
        summary["throttle_thermal_pct"] = pct(THERMAL_REASONS)
        summary["throttle_hw_pct"] = pct(HW_SLOWDOWN)
        return summary


def build_command(N, iterations, warmup):
    """shell command for one run, with module load if the module system is available"""
    run = f"{EXECUTABLE} {N} {iterations}"
    if warmup is not None:
        run += f" {warmup}"
    # `module` is a shell function defined by login scripts: check it inside a login shell
    return f"if command -v module > /dev/null 2>&1; then module load {MODULE}; fi; {run}"


def run_one(N, iterations, warmup):
    """run the executable once, return dict TIMING_COLUMNS -> value"""
    result = subprocess.run(["bash", "-lc", build_command(N, iterations, warmup)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Run with N={N} failed (exit code {result.returncode}):\n{result.stdout}{result.stderr}")

    matches = {
        "avg_time_us": avg_time_re(LABEL_TRANSFER).search(result.stdout),
        "gflops": gflops_re(LABEL_TRANSFER).search(result.stdout),
        "compute_time_us": avg_time_re(LABEL_COMPUTE).search(result.stdout),
        "compute_gflops": gflops_re(LABEL_COMPUTE).search(result.stdout),
    }
    missing = [name for name, m in matches.items() if m is None]
    if missing:
        sys.exit(f"Could not parse {missing} from the output for N={N}:\n{result.stdout}")
    return {name: float(m.group(1)) for name, m in matches.items()}


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
                        help="matrix sizes N (default: 500 1000 ... 15000)")
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
                timing = run_one(N, args.iterations, args.warmup)
                gpu = monitor.stop() if monitor else EMPTY_MONITOR
                writer.writerow([N, args.iterations] + [timing[c] for c in TIMING_COLUMNS]
                                + [gpu[c] for c in MONITOR_COLUMNS])
                line = (f"N = {N:6d}   with transfer: {timing['avg_time_us']:12.1f} us {timing['gflops']:9.1f} GFLOP/s"
                        f"   compute only: {timing['compute_time_us']:12.1f} us {timing['compute_gflops']:9.1f} GFLOP/s")
                if monitor:
                    line += (f"   active samples {gpu['active_samples']}"
                             f"   SM clock avg {gpu['sm_clock_avg_mhz']} MHz (min {gpu['sm_clock_min_mhz']})"
                             f"   {gpu['temp_max_c']} C   {gpu['power_avg_w']} W (limit {gpu['power_limit_w']} W)"
                             f"   throttle % power {gpu['throttle_power_pct']}"
                             f" thermal {gpu['throttle_thermal_pct']} hw {gpu['throttle_hw_pct']}")
                print(line)
    finally:
        if monitor:
            pynvml.nvmlShutdown()

    print(f"Results saved in {args.output}")


if __name__ == "__main__":
    main()
