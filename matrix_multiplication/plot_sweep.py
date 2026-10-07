#!/usr/bin/env python3
"""
Read the CSV produced by run_sweep.py and create two plots:
- GFLOP/s vs matrix size N
- average time per iteration vs matrix size N

usage:
    python3 plot_sweep.py                                       # reads sweep_results.csv
    python3 plot_sweep.py --input results.csv --prefix laptop   # laptop_gflops.png, laptop_avg_time.png
"""

import argparse
import csv
import sys

import matplotlib.pyplot as plt

# colors: one series per chart, so a single hue; text and grid stay neutral
LINE_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def read_csv(path):
    """return lists N, avg time (ms), gflops from the sweep CSV"""
    N, avg_time_ms, gflops = [], [], []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                N.append(int(row["N"]))
                avg_time_ms.append(float(row["avg_time_us"]) / 1000.0)  # us -> ms
                gflops.append(float(row["gflops"]))
    except FileNotFoundError:
        sys.exit(f"{path} not found: run run_sweep.py first")
    if not N:
        sys.exit(f"{path} contains no data")
    return N, avg_time_ms, gflops


def line_plot(x, y, title, ylabel, value_format, filename):
    """single series line plot, value labelled only on the last point"""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(x, y, color=LINE_COLOR, linewidth=2, marker="o", markersize=6)

    # direct label on the last point only
    ax.annotate(value_format.format(y[-1]), (x[-1], y[-1]), textcoords="offset points",
                xytext=(0, 8), ha="center", color=TEXT_PRIMARY, fontsize=9)

    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=12)
    ax.set_xlabel("matrix size N", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)

    # recessive grid and axes
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot GFLOP/s and average time from the sweep CSV")
    parser.add_argument("--input", default="sweep_results.csv", help="CSV from run_sweep.py (default: sweep_results.csv)")
    parser.add_argument("--prefix", default="sweep", help="prefix of the PNG files (default: sweep)")
    args = parser.parse_args()

    N, avg_time_ms, gflops = read_csv(args.input)

    line_plot(N, gflops, "cuBLAS SGEMM performance (computation + data transfer)",
              "GFLOP/s", "{:.0f}", f"{args.prefix}_gflops.png")
    line_plot(N, avg_time_ms, "Average time per iteration (computation + data transfer)",
              "time [ms]", "{:.1f} ms", f"{args.prefix}_avg_time.png")


if __name__ == "__main__":
    main()
