#!/usr/bin/env python3
"""
Read the CSV produced by run_sweep.py and create, as a function of the matrix size N:
- <prefix>_gflops.png and <prefix>_avg_time.png: GFLOP/s and average time, one figure each
- <prefix>_overview.png: one figure with a panel per logged quantity:
  GFLOP/s with transfer and computation only, average time, maximum temperature,
  SM clock average and minimum, average power.
  Monitor values can be missing for some N (short runs with no active GPU sample): those points
  are skipped; a panel with no data at all (e.g. sweep run with --no-monitor) is not drawn.

The helpers here (PANELS, read_csv, style_axes, overview_grid) are also used by compare_sweeps.py.

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
    ("compute_gflops",   "Performance (computation only)",            "GFLOP/s",   1.0,   "{:.0f}"),
    ("avg_time_us",      "Average time per iteration (+ transfer)",   "time [ms]", 1e-3,  "{:.1f} ms"),
    ("temp_max_c",       "GPU temperature, maximum",                  "°C",        1.0,   "{:.0f} °C"),
    ("sm_clock_avg_mhz", "SM clock, average (GPU active)",            "MHz",       1.0,   "{:.0f} MHz"),
    ("sm_clock_min_mhz", "SM clock, minimum (GPU active)",            "MHz",       1.0,   "{:.0f} MHz"),
    ("power_avg_w",      "GPU power, average (GPU active)",           "W",         1.0,   "{:.1f} W"),
]
N_COLS = 2
# panels drawn with the same y scale, so they can be compared at a glance
SAME_SCALE = ("gflops", "compute_gflops")


def read_csv(path):
    """return list N and dict column -> list of values (None where the cell is empty or missing)"""
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


def valid_points(x, y):
    """drop the points where y is missing"""
    pairs = [(a, b) for a, b in zip(x, y) if b is not None]
    return [a for a, _ in pairs], [b for _, b in pairs]


def style_axes(ax, title, ylabel, ymax):
    """common look: title, y from 0, recessive grid and axes"""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_ylim(bottom=0, top=ymax * 1.12)  # headroom for value labels
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def draw_panel(ax, x, y, title, ylabel, value_format):
    """single series line panel, value labelled only on the last point"""
    x, y = valid_points(x, y)
    ax.plot(x, y, color=LINE_COLOR, linewidth=2, marker="o", markersize=4)
    ax.annotate(value_format.format(y[-1]), (x[-1], y[-1]), textcoords="offset points",
                xytext=(0, 8), ha="center", color=TEXT_PRIMARY, fontsize=8)
    style_axes(ax, title, ylabel, max(y))


def overview_grid(n_panels):
    """figure with n_panels panels on N_COLS columns, shared x axis; returns fig and the used axes"""
    n_rows = math.ceil(n_panels / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(11, 3.2 * n_rows),
                             sharex=True, facecolor=SURFACE, squeeze=False)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)  # empty slot when the number of panels is odd
    axes = axes[:n_panels]
    # shared x axis: at most ~10 readable ticks, tick labels everywhere, axis label on the bottom row
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    for ax in axes:
        ax.tick_params(labelbottom=True)
    for ax in axes[-N_COLS:]:
        ax.set_xlabel("matrix size N", color=TEXT_SECONDARY)
    return fig, axes


def share_y_scale(axes, panels):
    """give the SAME_SCALE panels (if drawn) the same y range: from 0 to the largest top"""
    same = [ax for ax, p in zip(axes, panels) if p[0] in SAME_SCALE]
    if len(same) > 1:
        top = max(ax.get_ylim()[1] for ax in same)
        for ax in same:
            ax.set_ylim(0, top)


def save(fig, filename, top=1.0):
    """top < 1 leaves room above the panels (e.g. for a figure legend)"""
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def single_plot(N, y, title, ylabel, value_format, filename):
    """one quantity in its own figure"""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    draw_panel(ax, N, y, title, ylabel, value_format)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=12)
    ax.set_xlabel("matrix size N", color=TEXT_SECONDARY)
    # at most ~10 readable ticks, whatever the number of points
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    save(fig, filename)


def main():
    parser = argparse.ArgumentParser(description="Plot all logged quantities from the sweep CSV")
    parser.add_argument("--input", default="sweep_results.csv", help="CSV from run_sweep.py (default: sweep_results.csv)")
    parser.add_argument("--prefix", default="sweep", help="prefix of the PNG files (default: sweep)")
    args = parser.parse_args()

    N, columns = read_csv(args.input)

    # keep only panels with at least one value (monitor columns may be empty)
    panels = [p for p in PANELS if any(v is not None for v in columns[p[0]])]

    fig, axes = overview_grid(len(panels))
    for ax, (name, title, ylabel, _, value_format) in zip(axes, panels):
        draw_panel(ax, N, columns[name], title, ylabel, value_format)
    share_y_scale(axes, panels)
    fig.suptitle(f"cuBLAS SGEMM sweep ({args.input})", color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=12)
    save(fig, f"{args.prefix}_overview.png")

    # GFLOP/s and average time also as separate figures
    single_plot(N, columns["gflops"], "cuBLAS SGEMM performance (computation + data transfer)",
                "GFLOP/s", "{:.0f}", f"{args.prefix}_gflops.png")
    single_plot(N, columns["avg_time_us"], "Average time per iteration (computation + data transfer)",
                "time [ms]", "{:.1f} ms", f"{args.prefix}_avg_time.png")


if __name__ == "__main__":
    main()
