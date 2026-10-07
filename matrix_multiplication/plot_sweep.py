#!/usr/bin/env python3
"""
Read the CSV produced by run_sweep.py and create, as a function of the matrix size N:
- <prefix>_gflops.png and <prefix>_avg_time.png: GFLOP/s and average time, one figure each
- <prefix>_overview.png: one figure with a panel per logged quantity:
- GFLOP/s and average time per iteration (always present)
- SM clock average and minimum, maximum temperature, average power (GPU monitor columns;
  panels are skipped if the CSV has no monitor data, e.g. run with --no-monitor)
The throttle column is text, not a number: it is not plotted.

usage:
    python3 plot_sweep.py                                       # reads sweep_results.csv -> sweep_*.png
    python3 plot_sweep.py --input results.csv --prefix laptop   # -> laptop_overview.png, laptop_gflops.png, ...
"""

import argparse
import csv
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# colors: one series per panel, so a single hue; text and grid stay neutral
LINE_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"

# panels in order: (CSV column, panel title, y label, scale factor, format of the last value)
PANELS = [
    ("gflops",           "Performance (computation + data transfer)", "GFLOP/s",   1.0,   "{:.0f}"),
    ("avg_time_us",      "Average time per iteration",                "time [ms]", 1e-3,  "{:.1f} ms"),
    ("sm_clock_avg_mhz", "SM clock, average",                         "MHz",       1.0,   "{:.0f} MHz"),
    ("sm_clock_min_mhz", "SM clock, minimum",                         "MHz",       1.0,   "{:.0f} MHz"),
    ("temp_max_c",       "GPU temperature, maximum",                  "°C",        1.0,   "{:.0f} °C"),
    ("power_avg_w",      "GPU power, average",                        "W",         1.0,   "{:.1f} W"),
]
N_COLS = 2


def read_csv(path):
    """return list N and dict column -> list of values (None where the cell is empty)"""
    N = []
    columns = {name: [] for name, *_ in PANELS}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                N.append(int(row["N"]))
                for name, _, _, scale, _ in PANELS:
                    cell = row.get(name, "")
                    columns[name].append(float(cell) * scale if cell not in ("", None) else None)
    except FileNotFoundError:
        sys.exit(f"{path} not found: run run_sweep.py first")
    if not N:
        sys.exit(f"{path} contains no data")
    return N, columns


def draw_panel(ax, x, y, title, ylabel, value_format):
    """single series line panel, value labelled only on the last point"""
    ax.set_facecolor(SURFACE)
    ax.plot(x, y, color=LINE_COLOR, linewidth=2, marker="o", markersize=4)

    # direct label on the last point only
    ax.annotate(value_format.format(y[-1]), (x[-1], y[-1]), textcoords="offset points",
                xytext=(0, 8), ha="center", color=TEXT_PRIMARY, fontsize=8)

    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_ylim(bottom=0, top=max(y) * 1.12)  # headroom for the last value label

    # recessive grid and axes
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def single_plot(N, y, title, ylabel, value_format, filename):
    """one quantity in its own figure"""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    draw_panel(ax, N, y, title, ylabel, value_format)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=12)
    ax.set_xlabel("matrix size N", color=TEXT_SECONDARY)
    # at most ~10 readable ticks, whatever the number of points
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot all logged quantities from the sweep CSV in one figure")
    parser.add_argument("--input", default="sweep_results.csv", help="CSV from run_sweep.py (default: sweep_results.csv)")
    parser.add_argument("--prefix", default="sweep", help="prefix of the PNG file (default: sweep)")
    args = parser.parse_args()

    N, columns = read_csv(args.input)

    # keep only panels with data in every row (monitor columns may be empty)
    panels = [p for p in PANELS if all(v is not None for v in columns[p[0]])]

    n_rows = math.ceil(len(panels) / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(11, 3.2 * n_rows),
                             sharex=True, facecolor=SURFACE, squeeze=False)
    axes = axes.flatten()

    for ax, (name, title, ylabel, _, value_format) in zip(axes, panels):
        draw_panel(ax, N, columns[name], title, ylabel, value_format)
    for ax in axes[len(panels):]:
        ax.set_visible(False)  # empty slot when the number of panels is odd

    # shared x axis: at most ~10 readable ticks, label on the bottom row only
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    for ax in axes[:len(panels)]:
        ax.tick_params(labelbottom=True)
    for ax in axes[len(panels) - N_COLS:len(panels)]:
        ax.set_xlabel("matrix size N", color=TEXT_SECONDARY)

    fig.suptitle(f"cuBLAS SGEMM sweep ({args.input})", color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    filename = f"{args.prefix}_overview.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")

    # GFLOP/s and average time also as separate figures
    single_plot(N, columns["gflops"], "cuBLAS SGEMM performance (computation + data transfer)",
                "GFLOP/s", "{:.0f}", f"{args.prefix}_gflops.png")
    single_plot(N, columns["avg_time_us"], "Average time per iteration (computation + data transfer)",
                "time [ms]", "{:.1f} ms", f"{args.prefix}_avg_time.png")


if __name__ == "__main__":
    main()
