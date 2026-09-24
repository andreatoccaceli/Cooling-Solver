#!/usr/bin/env python3
"""Plot OpenMP and OpenACC CPU strong-scaling benchmark results.

The script groups repeated measurements by implementation and CPU thread
count. Lines show the median runtime for each configuration.

Speedup is computed independently for each implementation using its median
one-thread runtime as the baseline::

    S(p) = median(T(1)) / median(T(p))

"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "logs" / "benchmark.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

DEVICE_LABELS = {
    "omp": "OpenMP",
    "acc_cpu": "OpenACC CPU",
}
DEVICE_COLORS = {
    "omp": "#0072B2",
    "acc_cpu": "#D55E00",
}
DEVICE_MARKERS = {
    "omp": "o",
    "acc_cpu": "s",
}

METRICS = {
    "pure-dynamics": (
        "Pure dynamics compute time",
        "Pure dynamics time (s)",
        "pure_dynamics",
    ),
    "total": (
        "Total measured wall time",
        "Measured solver time (s)",
        "total_time",
    ),
}


@dataclass(frozen=True)
class Summary:
    device: str
    threads: int
    values: np.ndarray
    median: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Benchmark CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRICS),
        default="pure-dynamics",
        help="Timing used for scaling. Default: pure-dynamics.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Raster output resolution. Default: 220.",
    )
    return parser.parse_args()


def load_measurements(
    path: Path,
    metric_column: str,
) -> tuple[dict[tuple[str, int], list[float]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    problem_signatures: set[tuple[str, str, str]] = set()

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "Device",
            "Cpu threads",
            "Grid",
            "Max fractal iterations",
            "Time steps",
            metric_column,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Benchmark CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            device = row["Device"].strip()
            if device not in DEVICE_LABELS:
                continue

            try:
                threads = int(row["Cpu threads"])
                value = float(row[metric_column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid CPU benchmark value on CSV line {line_number}"
                ) from error

            if threads <= 0 or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Non-positive or non-finite value on CSV line {line_number}"
                )

            grouped[(device, threads)].append(value)
            problem_signatures.add(
                (
                    row["Grid"].strip(),
                    row["Max fractal iterations"].strip(),
                    row["Time steps"].strip(),
                )
            )

    if not grouped:
        raise ValueError("No OpenMP or OpenACC CPU measurements found")
    if len(problem_signatures) != 1:
        raise ValueError(
            "CPU rows contain multiple problem configurations; "
            "filter the CSV before plotting"
        )

    grid, fractal_iterations, time_steps = next(iter(problem_signatures))
    problem_description = (
        f"Grid {grid}, {time_steps} time steps, "
        f"{fractal_iterations} max fractal iterations"
    )
    return dict(grouped), problem_description


def summarize(
    grouped: dict[tuple[str, int], list[float]],
) -> dict[str, list[Summary]]:
    summaries: dict[str, list[Summary]] = {}

    for device in DEVICE_LABELS:
        device_rows: list[Summary] = []
        for (row_device, threads), raw_values in grouped.items():
            if row_device != device:
                continue

            values = np.asarray(raw_values, dtype=np.float64)
            median = np.median(values)
            device_rows.append(
                Summary(
                    device=device,
                    threads=threads,
                    values=values,
                    median=float(median),
                )
            )

        device_rows.sort(key=lambda row: row.threads)
        if not device_rows:
            raise ValueError(f"No measurements found for {DEVICE_LABELS[device]}")
        if device_rows[0].threads != 1:
            raise ValueError(
                f"{DEVICE_LABELS[device]} needs a one-thread baseline"
            )
        summaries[device] = device_rows

    thread_sets = {
        device: {row.threads for row in rows}
        for device, rows in summaries.items()
    }
    if len({frozenset(values) for values in thread_sets.values()}) != 1:
        raise ValueError(
            "OpenMP and OpenACC CPU must contain the same thread counts: "
            + "; ".join(
                f"{DEVICE_LABELS[key]}={sorted(values)}"
                for key, values in thread_sets.items()
            )
        )

    return summaries


def configure_scaling_axis(ax: plt.Axes, threads: Sequence[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("CPU threads")
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)


def add_runtime_series(
    ax: plt.Axes,
    rows: Sequence[Summary],
) -> None:
    device = rows[0].device
    color = DEVICE_COLORS[device]
    marker = DEVICE_MARKERS[device]
    threads = np.asarray([row.threads for row in rows], dtype=np.float64)
    medians = np.asarray([row.median for row in rows])
    ax.plot(
        threads,
        medians,
        color=color,
        marker=marker,
        markersize=4.5,
        linewidth=2,
        label=DEVICE_LABELS[device],
        zorder=3,
    )


def add_speedup_series(ax: plt.Axes, rows: Sequence[Summary]) -> None:
    device = rows[0].device
    baseline = rows[0].median
    threads = np.asarray([row.threads for row in rows], dtype=np.float64)
    speedup = np.asarray([baseline / row.median for row in rows])
    ax.plot(
        threads,
        speedup,
        color=DEVICE_COLORS[device],
        marker=DEVICE_MARKERS[device],
        markersize=4.5,
        linewidth=2,
        label=DEVICE_LABELS[device],
        zorder=3,
    )


def create_figure(
    summaries: dict[str, list[Summary]],
    metric_label: str,
    problem_description: str,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )

    fig, (runtime_ax, speedup_ax) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.8),
    )
    threads = [row.threads for row in summaries["omp"]]

    for device in DEVICE_LABELS:
        add_runtime_series(runtime_ax, summaries[device])
        add_speedup_series(speedup_ax, summaries[device])

    configure_scaling_axis(runtime_ax, threads)
    runtime_ax.set_yscale("log", base=2)
    runtime_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}")
    )
    runtime_ax.set_ylabel(metric_label)
    runtime_ax.set_title("Runtime")
    runtime_ax.legend(frameon=False)

    configure_scaling_axis(speedup_ax, threads)
    ideal_threads = np.asarray(threads, dtype=np.float64)
    speedup_ax.plot(
        ideal_threads,
        ideal_threads,
        linestyle="--",
        color="#666666",
        linewidth=1.4,
        label="Ideal",
        zorder=1,
    )
    speedup_ax.set_yscale("log", base=2)
    speedup_ax.set_yticks(threads)
    speedup_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}")
    )
    speedup_ax.set_ylabel("Speedup relative to 1 thread")
    speedup_ax.set_title("Strong scaling")
    speedup_ax.legend(frameon=False)

    repetitions = sorted(
        {
            len(row.values)
            for rows in summaries.values()
            for row in rows
        }
    )
    repetition_text = (
        f"n={repetitions[0]} per configuration"
        if len(repetitions) == 1
        else f"n={min(repetitions)}--{max(repetitions)} per configuration"
    )
    fig.suptitle("CPU strong scaling")
    fig.text(
        0.5,
        0.025,
        f"{problem_description}. Median; {repetition_text}.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    return fig


def print_summary(summaries: dict[str, list[Summary]]) -> None:
    print("Implementation  Threads  Runs  Median (s)")
    for device in DEVICE_LABELS:
        for row in summaries[device]:
            print(
                f"{DEVICE_LABELS[device]:<15} {row.threads:>7} "
                f"{len(row.values):>5}  "
                f"{row.median:.6f}"
            )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    metric_column, metric_label, metric_stem = METRICS[args.metric]
    grouped, problem_description = load_measurements(
        args.input,
        metric_column,
    )
    summaries = summarize(grouped)
    print_summary(summaries)

    fig = create_figure(summaries, metric_label, problem_description)
    output = save_figure(
        fig,
        args.output_dir,
        f"cpu_scaling_{metric_stem}",
        args.dpi,
    )
    plt.close(fig)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
