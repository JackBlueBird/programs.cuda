#!/usr/bin/env python3
"""
Compare two CSV files produced by run_sweep.py (e.g. laptop in power saver vs performance mode):
one figure with the same panels of plot_sweep.py, each with both sweeps as two lines.

usage:
    python3 compare_sweeps.py powersaver.csv performance.csv
    python3 compare_sweeps.py a.csv b.csv --labels "power saver" "performance" --output compare.png
"""

import argparse
from pathlib import Path

from plot_sweep import (PANELS, TEXT_PRIMARY, overview_grid, read_csv, save, share_y_scale,
                        style_axes, valid_points)

# two series: fixed order of the categorical palette (blue, orange) + different marker shapes,
# so the sweeps are told apart also without color
SERIES_STYLE = [
    {"color": "#2a78d6", "marker": "o"},
    {"color": "#eb6834", "marker": "s"},
]


def main():
    parser = argparse.ArgumentParser(description="Compare two sweep CSV files in one multi-panel figure")
    parser.add_argument("csv_a", help="first CSV from run_sweep.py")
    parser.add_argument("csv_b", help="second CSV from run_sweep.py")
    parser.add_argument("--labels", nargs=2, metavar=("LABEL_A", "LABEL_B"),
                        help="legend names (default: the file names)")
    parser.add_argument("--output", default="compare_overview.png", help="output PNG (default: compare_overview.png)")
    args = parser.parse_args()

    paths = [args.csv_a, args.csv_b]
    labels = args.labels or [Path(p).stem for p in paths]
    sweeps = [read_csv(p) for p in paths]  # each: (N list, columns dict)

    # keep the panels where at least one of the two sweeps has data
    panels = [p for p in PANELS if any(v is not None for _, cols in sweeps for v in cols[p[0]])]

    fig, axes = overview_grid(len(panels))
    for ax, (name, title, ylabel, _, value_format) in zip(axes, panels):
        series = []  # (x, y) of each sweep with data in this panel
        for (N, cols), label, style in zip(sweeps, labels, SERIES_STYLE):
            x, y = valid_points(N, cols[name])
            if not y:
                continue
            ax.plot(x, y, color=style["color"], marker=style["marker"], markersize=4, linewidth=2, label=label)
            series.append((x, y))

        # value of the last point of each series: the higher value labelled above its point,
        # the lower one below, so the labels keep the same vertical order as the lines
        # (rank by value, not by equality, so two equal values still get one label above and one below)
        for rank, (x, y) in enumerate(sorted(series, key=lambda s: s[1][-1], reverse=True)):
            offset = 8 if rank == 0 else -14
            ax.annotate(value_format.format(y[-1]), (x[-1], y[-1]), textcoords="offset points",
                        xytext=(0, offset), ha="center", color=TEXT_PRIMARY, fontsize=8)
        style_axes(ax, title, ylabel, max(max(y) for _, y in series))
    share_y_scale(axes, panels)

    # one legend for the whole figure, above the panels
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("cuBLAS SGEMM sweep comparison", color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=12)
    save(fig, args.output, top=0.97)


if __name__ == "__main__":
    main()
