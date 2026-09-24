#!/usr/bin/env python3
"""Generate the complete Cooling Solver benchmark analysis as PNG figures.

The input CSV is expected to contain repeated benchmark measurements for the
serial C++, OpenMP, OpenACC CPU, OpenACC GPU, and CUDA implementations. Every
figure uses the median of the repeated runs. CPU parameter sweeps use the
32-thread measurements, while CPU scaling uses all available thread counts.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "logs" / "benchmark.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

BASELINE_GRID = (4000, 4000)
BASELINE_ITERATIONS = 4000
BASELINE_STEPS = 200
CPU_THREADS_FOR_COMPARISON = 32
EXPECTED_REPETITIONS = 5

DEVICE_ORDER = ("cpp", "omp", "acc_cpu", "acc_gpu", "cuda")
CPU_DEVICES = ("omp", "acc_cpu")
DEVICE_LABELS = {
    "cpp": "C++ serial",
    "omp": "OpenMP",
    "acc_cpu": "OpenACC CPU",
    "acc_gpu": "OpenACC GPU",
    "cuda": "CUDA",
}
DEVICE_COLORS = {
    "cpp": "#555555",
    "omp": "#0072B2",
    "acc_cpu": "#D55E00",
    "acc_gpu": "#009E73",
    "cuda": "#CC79A7",
}
DEVICE_MARKERS = {
    "cpp": "D",
    "omp": "o",
    "acc_cpu": "s",
    "acc_gpu": "^",
    "cuda": "v",
}

METRIC_LABELS = {
    "Weight field time": "Weight field time (s)",
    "Weight range reduction time": "Weight range time (s)",
    "Initialization time": "Initialization time (s)",
    "Pure dynamics compute time": "Pure dynamics time (s)",
    "Statistics time": "Statistics time (s)",
    "Total measured wall time": "Measured solver time (s)",
    "Pure dynamics performance": "Pure dynamics performance (GLUP/s)",
}
NUMERIC_METRICS = tuple(METRIC_LABELS)
GRID_PATTERN = re.compile(r"^(\d+)\s*x\s*(\d+)$")

PLOT_CHOICES = (
    "cpu-scaling",
    "cpu-efficiency",
    "backend",
    "phases",
    "grid",
    "grid-speedup",
    "iterations",
    "iterations-speedup",
    "steps",
    "steps-speedup",
    "variability",
)


@dataclass(frozen=True)
class BenchmarkRow:
    device: str
    cpu_threads: int | None
    grid_width: int
    grid_height: int
    max_iterations: int
    time_steps: int
    metrics: dict[str, float]


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
        help=f"Directory for PNG figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("all", *PLOT_CHOICES),
        default=("all",),
        help="Figures to generate. Default: all.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution. Default: 220.",
    )
    return parser.parse_args()


def parse_positive_int(raw_value: str, field: str, line_number: int) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Invalid {field} on CSV line {line_number}: {raw_value!r}"
        ) from error
    if value <= 0:
        raise ValueError(
            f"Non-positive {field} on CSV line {line_number}: {value}"
        )
    return value


def load_rows(path: Path) -> list[BenchmarkRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")

    required = {
        "Device",
        "Cpu threads",
        "Grid",
        "Max fractal iterations",
        "Time steps",
        *NUMERIC_METRICS,
    }
    rows: list[BenchmarkRow] = []

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Benchmark CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        for line_number, raw in enumerate(reader, start=2):
            device = raw["Device"].strip()
            if device not in DEVICE_LABELS:
                raise ValueError(
                    f"Unknown device on CSV line {line_number}: {device!r}"
                )

            match = GRID_PATTERN.fullmatch(raw["Grid"].strip())
            if match is None:
                raise ValueError(
                    f"Invalid Grid on CSV line {line_number}: {raw['Grid']!r}"
                )
            grid_width, grid_height = map(int, match.groups())

            threads_raw = raw["Cpu threads"].strip()
            cpu_threads = (
                parse_positive_int(threads_raw, "Cpu threads", line_number)
                if threads_raw
                else None
            )
            if device in CPU_DEVICES and cpu_threads is None:
                raise ValueError(
                    f"Missing Cpu threads for {device} on CSV line {line_number}"
                )

            metrics: dict[str, float] = {}
            for metric in NUMERIC_METRICS:
                try:
                    value = float(raw[metric])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid {metric} on CSV line {line_number}"
                    ) from error
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"Invalid {metric} on CSV line {line_number}: {value}"
                    )
                metrics[metric] = value

            rows.append(
                BenchmarkRow(
                    device=device,
                    cpu_threads=cpu_threads,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    max_iterations=parse_positive_int(
                        raw["Max fractal iterations"],
                        "Max fractal iterations",
                        line_number,
                    ),
                    time_steps=parse_positive_int(
                        raw["Time steps"], "Time steps", line_number
                    ),
                    metrics=metrics,
                )
            )

    if not rows:
        raise ValueError("Benchmark CSV contains no rows")
    return rows


def is_baseline(row: BenchmarkRow) -> bool:
    return (
        (row.grid_width, row.grid_height) == BASELINE_GRID
        and row.max_iterations == BASELINE_ITERATIONS
        and row.time_steps == BASELINE_STEPS
    )


def uses_comparison_resources(row: BenchmarkRow) -> bool:
    return row.device not in CPU_DEVICES or row.cpu_threads == CPU_THREADS_FOR_COMPARISON


def median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def group_rows(
    rows: Sequence[BenchmarkRow],
    key: Callable[[BenchmarkRow], object],
) -> dict[object, list[BenchmarkRow]]:
    grouped: dict[object, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return dict(grouped)


def require_repetitions(
    grouped: dict[object, list[BenchmarkRow]],
    context: str,
) -> None:
    bad = {
        str(key): len(group)
        for key, group in grouped.items()
        if len(group) != EXPECTED_REPETITIONS
    }
    if bad:
        details = ", ".join(f"{key}: n={count}" for key, count in bad.items())
        raise ValueError(
            f"{context} requires n={EXPECTED_REPETITIONS} per group; {details}"
        )


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 4.5,
        }
    )


def configure_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)


def configure_log_x(ax: plt.Axes, values: Sequence[int], label: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    if len(values) >= 7:
        ax.tick_params(axis="x", labelrotation=25)
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")
    ax.set_xlabel(label)
    configure_axis(ax)


def configure_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    configure_axis(ax)


def add_figure_note(fig: plt.Figure, text: str, bottom: float = 0.025) -> None:
    fig.text(
        0.5,
        bottom,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )


def add_shared_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncols=len(labels),
        frameon=False,
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
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def cpu_scaling_figure(
    rows: Sequence[BenchmarkRow],
    metric: str,
    title: str,
) -> plt.Figure:
    selected = [
        row for row in rows if row.device in CPU_DEVICES and is_baseline(row)
    ]
    grouped = group_rows(selected, lambda row: (row.device, row.cpu_threads))
    require_repetitions(grouped, title)

    thread_sets = {
        device: sorted(
            int(thread)
            for row_device, thread in grouped
            if row_device == device and thread is not None
        )
        for device in CPU_DEVICES
    }
    if thread_sets["omp"] != thread_sets["acc_cpu"]:
        raise ValueError(f"CPU scaling thread sets differ: {thread_sets}")
    threads = thread_sets["omp"]
    if not threads or threads[0] != 1:
        raise ValueError("CPU scaling requires a one-thread baseline")

    fig, (runtime_ax, speedup_ax) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for device in CPU_DEVICES:
        runtimes = np.asarray(
            [
                median(
                    [row.metrics[metric] for row in grouped[(device, thread)]]
                )
                for thread in threads
            ]
        )
        thread_array = np.asarray(threads, dtype=np.float64)
        runtime_ax.plot(
            thread_array,
            runtimes,
            color=DEVICE_COLORS[device],
            marker=DEVICE_MARKERS[device],
            label=DEVICE_LABELS[device],
        )
        speedup_ax.plot(
            thread_array,
            runtimes[0] / runtimes,
            color=DEVICE_COLORS[device],
            marker=DEVICE_MARKERS[device],
            label=DEVICE_LABELS[device],
        )

    configure_log_x(runtime_ax, threads, "CPU threads")
    runtime_ax.set_yscale("log", base=2)
    runtime_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}")
    )
    runtime_ax.set_ylabel(METRIC_LABELS[metric])
    runtime_ax.set_title("Runtime")
    runtime_ax.legend(frameon=False)

    configure_log_x(speedup_ax, threads, "CPU threads")
    speedup_ax.plot(
        threads,
        threads,
        linestyle="--",
        color="#666666",
        linewidth=1.4,
        label="Ideal",
    )
    speedup_ax.set_yscale("log", base=2)
    speedup_ax.set_yticks(threads)
    speedup_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}")
    )
    speedup_ax.set_ylabel("Speedup relative to 1 thread")
    speedup_ax.set_title("Strong scaling")
    speedup_ax.legend(frameon=False)

    fig.suptitle(title)
    add_figure_note(
        fig,
        "Grid 4000 x 4000, 200 time steps, 4000 max fractal iterations. "
        "Median; n=5 per configuration.",
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    return fig


def cpu_efficiency_figure(rows: Sequence[BenchmarkRow]) -> plt.Figure:
    selected = [
        row for row in rows if row.device in CPU_DEVICES and is_baseline(row)
    ]
    grouped = group_rows(selected, lambda row: (row.device, row.cpu_threads))
    require_repetitions(grouped, "CPU parallel efficiency")

    thread_sets = {
        device: sorted(
            int(thread)
            for row_device, thread in grouped
            if row_device == device and thread is not None
        )
        for device in CPU_DEVICES
    }
    if thread_sets["omp"] != thread_sets["acc_cpu"]:
        raise ValueError(f"CPU efficiency thread sets differ: {thread_sets}")
    threads = thread_sets["omp"]
    if not threads or threads[0] != 1:
        raise ValueError("CPU efficiency requires a one-thread baseline")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    specifications = (
        ("Pure dynamics compute time", "Pure dynamics"),
        ("Total measured wall time", "Complete measured solver"),
    )
    thread_array = np.asarray(threads, dtype=np.float64)

    for ax, (metric, title) in zip(axes, specifications):
        for device in CPU_DEVICES:
            runtimes = np.asarray(
                [
                    median(
                        [row.metrics[metric] for row in grouped[(device, thread)]]
                    )
                    for thread in threads
                ]
            )
            efficiency = 100.0 * (runtimes[0] / runtimes) / thread_array
            ax.plot(
                thread_array,
                efficiency,
                color=DEVICE_COLORS[device],
                marker=DEVICE_MARKERS[device],
                label=DEVICE_LABELS[device],
            )
        ax.axhline(
            100.0,
            linestyle="--",
            color="#666666",
            linewidth=1.4,
            label="Ideal",
        )
        configure_log_x(ax, threads, "CPU threads")
        ax.set_ylim(0.0, 108.0)
        ax.set_ylabel("Parallel efficiency (%)")
        ax.set_title(title)

    fig.suptitle("CPU strong-scaling efficiency")
    add_shared_legend(fig, axes[0])
    add_figure_note(
        fig,
        "Efficiency E(p) = T(1) / [p T(p)]. Grid 4000 x 4000, "
        "200 time steps, 4000 max fractal iterations. Median; n=5.",
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.87))
    return fig


def baseline_groups(
    rows: Sequence[BenchmarkRow],
    context: str,
) -> dict[object, list[BenchmarkRow]]:
    selected = [
        row for row in rows if is_baseline(row) and uses_comparison_resources(row)
    ]
    grouped = group_rows(selected, lambda row: row.device)
    missing = set(DEVICE_ORDER).difference(grouped)
    if missing:
        raise ValueError(f"{context} is missing devices: {sorted(missing)}")
    require_repetitions(grouped, context)
    return grouped


def annotate_bars(ax: plt.Axes, bars: Sequence[plt.Rectangle]) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.3g}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def add_bar_label_headroom(ax: plt.Axes, values: Sequence[float]) -> None:
    """Reserve space above bars for value labels, including on logarithmic axes."""
    lower_limit, _ = ax.get_ylim()
    ax.set_ylim(lower_limit, max(values) * 1.8)


def backend_comparison_figure(rows: Sequence[BenchmarkRow]) -> plt.Figure:
    grouped = baseline_groups(rows, "Backend comparison")
    fig, (performance_ax, total_ax) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    x = np.arange(len(DEVICE_ORDER))
    colors = [DEVICE_COLORS[device] for device in DEVICE_ORDER]
    labels = [DEVICE_LABELS[device] for device in DEVICE_ORDER]

    performance = [
        median([row.metrics["Pure dynamics performance"] for row in grouped[device]])
        for device in DEVICE_ORDER
    ]
    bars = performance_ax.bar(x, performance, color=colors, width=0.68)
    performance_ax.set_xticks(x, labels, rotation=18, ha="right")
    performance_ax.set_ylabel("Pure dynamics performance (GLUP/s)")
    performance_ax.set_title("Stencil throughput")
    configure_log_y(performance_ax)
    annotate_bars(performance_ax, bars)
    add_bar_label_headroom(performance_ax, performance)

    total_times = [
        median([row.metrics["Total measured wall time"] for row in grouped[device]])
        for device in DEVICE_ORDER
    ]
    bars = total_ax.bar(x, total_times, color=colors, width=0.68)
    total_ax.set_xticks(x, labels, rotation=18, ha="right")
    total_ax.set_ylabel("Measured solver time (s)")
    total_ax.set_title("Complete measured solver")
    configure_log_y(total_ax)
    annotate_bars(total_ax, bars)
    add_bar_label_headroom(total_ax, total_times)

    fig.suptitle("Backend performance at the reference configuration")
    add_figure_note(
        fig,
        "Grid 4000 x 4000, 200 time steps, 4000 max fractal iterations. "
        "CPU parallel backends use 32 threads. Median; n=5.",
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    return fig


def phase_breakdown_figure(rows: Sequence[BenchmarkRow]) -> plt.Figure:
    grouped = baseline_groups(rows, "Phase breakdown")
    phases = (
        "Weight field time",
        "Weight range reduction time",
        "Initialization time",
        "Pure dynamics compute time",
        "Statistics time",
    )
    phase_labels = ("Weights", "Range", "Initialization", "Dynamics", "Statistics")
    phase_colors = ("#56B4E9", "#E69F00", "#F0E442", "#0072B2", "#009E73")

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(len(DEVICE_ORDER), dtype=np.float64)
    width = 0.15
    offsets = (np.arange(len(phases)) - (len(phases) - 1) / 2.0) * width

    for offset, phase, label, color in zip(
        offsets, phases, phase_labels, phase_colors
    ):
        values = [
            median([row.metrics[phase] for row in grouped[device]])
            for device in DEVICE_ORDER
        ]
        ax.bar(x + offset, values, width=width, label=label, color=color)

    ax.set_xticks(x, [DEVICE_LABELS[device] for device in DEVICE_ORDER])
    ax.set_ylabel("Median time (s)")
    ax.set_title("Execution time by computational phase")
    configure_log_y(ax)
    ax.legend(frameon=False, ncols=3)
    add_figure_note(
        fig,
        "Reference configuration; CPU parallel backends use 32 threads. "
        "CSV and HDF5 times are excluded. Median; n=5.",
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    return fig


def sweep_groups(
    rows: Sequence[BenchmarkRow],
    kind: str,
) -> tuple[dict[object, list[BenchmarkRow]], list[int]]:
    selected: list[BenchmarkRow] = []
    value_for: Callable[[BenchmarkRow], int]

    if kind == "grid":
        selected = [
            row
            for row in rows
            if uses_comparison_resources(row)
            and row.grid_width == row.grid_height
            and row.max_iterations == BASELINE_ITERATIONS
            and row.time_steps == BASELINE_STEPS
        ]
        value_for = lambda row: row.grid_width
    elif kind == "iterations":
        selected = [
            row
            for row in rows
            if uses_comparison_resources(row)
            and (row.grid_width, row.grid_height) == BASELINE_GRID
            and row.time_steps == BASELINE_STEPS
        ]
        value_for = lambda row: row.max_iterations
    elif kind == "steps":
        selected = [
            row
            for row in rows
            if uses_comparison_resources(row)
            and (row.grid_width, row.grid_height) == BASELINE_GRID
            and row.max_iterations == BASELINE_ITERATIONS
        ]
        value_for = lambda row: row.time_steps
    else:
        raise ValueError(f"Unknown sweep: {kind}")

    grouped = group_rows(selected, lambda row: (row.device, value_for(row)))
    require_repetitions(grouped, f"{kind} sweep")
    value_sets = {
        device: {int(value) for row_device, value in grouped if row_device == device}
        for device in DEVICE_ORDER
    }
    if len({frozenset(values) for values in value_sets.values()}) != 1:
        raise ValueError(f"{kind} sweep values differ by device: {value_sets}")
    values = sorted(value_sets[DEVICE_ORDER[0]])
    return grouped, values


def add_sweep_metric(
    ax: plt.Axes,
    grouped: dict[object, list[BenchmarkRow]],
    values: Sequence[int],
    metric: str,
    title: str,
    x_label: str,
    *,
    log_y: bool = True,
) -> None:
    for device in DEVICE_ORDER:
        medians = [
            median([row.metrics[metric] for row in grouped[(device, value)]])
            for value in values
        ]
        ax.plot(
            values,
            medians,
            color=DEVICE_COLORS[device],
            marker=DEVICE_MARKERS[device],
            label=DEVICE_LABELS[device],
        )
    configure_log_x(ax, values, x_label)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(title)


def sensitivity_sweep_figure(
    rows: Sequence[BenchmarkRow],
    kind: str,
    x_label: str,
    figure_title: str,
    note: str,
) -> plt.Figure:
    """Build the common four-panel layout used by every sensitivity sweep."""
    grouped, values = sweep_groups(rows, kind)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    specifications = (
        ("Pure dynamics performance", "Stencil throughput"),
        ("Pure dynamics compute time", "Dynamics runtime"),
        ("Weight field time", "Fractal-weight runtime"),
        ("Total measured wall time", "Complete measured solver"),
    )
    for ax, (metric, title) in zip(axes.flat, specifications):
        add_sweep_metric(
            ax,
            grouped,
            values,
            metric,
            title,
            x_label,
        )
    fig.suptitle(figure_title)
    add_shared_legend(fig, axes[0, 0])
    add_figure_note(fig, note)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.88))
    return fig


def grid_sweep_figure(rows: Sequence[BenchmarkRow]) -> plt.Figure:
    return sensitivity_sweep_figure(
        rows,
        "grid",
        "Grid side length N",
        "Grid-size sensitivity",
        "Square grids; 200 time steps; 4000 max fractal iterations. "
        "CPU parallel backends use 32 threads. Median; n=5.",
    )


def sweep_speedup_figure(
    rows: Sequence[BenchmarkRow],
    kind: str,
    x_label: str,
    title: str,
    note: str,
) -> plt.Figure:
    """Plot total-time speedup over any supported parameter sweep."""
    grouped, values = sweep_groups(rows, kind)
    serial_times = np.asarray(
        [
            median(
                [
                    row.metrics["Total measured wall time"]
                    for row in grouped[("cpp", value)]
                ]
            )
            for value in values
        ]
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for device in DEVICE_ORDER:
        device_times = np.asarray(
            [
                median(
                    [
                        row.metrics["Total measured wall time"]
                        for row in grouped[(device, value)]
                    ]
                )
                for value in values
            ]
        )
        ax.plot(
            values,
            serial_times / device_times,
            color=DEVICE_COLORS[device],
            marker=DEVICE_MARKERS[device],
            label=DEVICE_LABELS[device],
        )

    configure_log_x(ax, values, x_label)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_ylabel("Speedup relative to C++ serial")
    ax.set_title(title)
    ax.legend(frameon=False, ncols=2)
    add_figure_note(fig, note)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    return fig


def iterations_sweep_figure(rows: Sequence[BenchmarkRow]) -> plt.Figure:
    return sensitivity_sweep_figure(
        rows,
        "iterations",
        "Maximum fractal iterations",
        "Fractal-iteration sensitivity",
        "Grid 4000 x 4000; 200 time steps. CPU parallel backends use "
        "32 threads. Median; n=5.",
    )


def steps_sweep_figure(rows: Sequence[BenchmarkRow]) -> plt.Figure:
    return sensitivity_sweep_figure(
        rows,
        "steps",
        "Time steps",
        "Time-step sensitivity",
        "Grid 4000 x 4000; 4000 max fractal iterations. CPU parallel "
        "backends use 32 threads. Median; n=5.",
    )


def style_table(table: object, header_color: str = "#D9EAF7") -> None:
    cells = table.get_celld()
    for (row, _column), cell in cells.items():
        cell.set_edgecolor("#B7B7B7")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")


def variability_figure(rows: Sequence[BenchmarkRow]) -> plt.Figure:
    metrics = (
        "Pure dynamics compute time",
        "Weight field time",
        "Total measured wall time",
        "Pure dynamics performance",
    )
    short_metric_names = {
        "Pure dynamics compute time": "Dynamics time",
        "Weight field time": "Weight-field time",
        "Total measured wall time": "Total time",
        "Pure dynamics performance": "Dynamics performance",
    }
    grouped = group_rows(
        rows,
        lambda row: (
            row.device,
            row.cpu_threads,
            row.grid_width,
            row.max_iterations,
            row.time_steps,
        ),
    )
    require_repetitions(grouped, "Variability analysis")

    records: list[dict[str, object]] = []
    for key, group in grouped.items():
        device, threads, grid_width, iterations, steps = key
        for metric in metrics:
            samples = np.asarray(
                [row.metrics[metric] for row in group], dtype=np.float64
            )
            q25, q50, q75 = np.percentile(samples, (25.0, 50.0, 75.0))
            relative_iqr = 100.0 * (q75 - q25) / q50 if q50 > 0.0 else 0.0
            records.append(
                {
                    "metric": metric,
                    "device": str(device),
                    "threads": threads,
                    "grid": int(grid_width),
                    "iterations": int(iterations),
                    "steps": int(steps),
                    "relative_iqr": relative_iqr,
                }
            )

    summary_rows: list[list[str]] = []
    for metric in metrics:
        values = np.asarray(
            [
                float(record["relative_iqr"])
                for record in records
                if record["metric"] == metric
            ]
        )
        summary_rows.append(
            [
                short_metric_names[metric],
                f"{np.median(values):.2f}%",
                f"{np.percentile(values, 95.0):.2f}%",
                f"{np.max(values):.2f}%",
            ]
        )

    most_variable = sorted(
        records,
        key=lambda record: float(record["relative_iqr"]),
        reverse=True,
    )[:10]
    detail_rows = []
    for rank, record in enumerate(most_variable, start=1):
        threads = record["threads"]
        detail_rows.append(
            [
                str(rank),
                short_metric_names[str(record["metric"])],
                DEVICE_LABELS[str(record["device"])],
                "-" if threads is None else str(int(threads)),
                f"{record['grid']}²",
                str(record["iterations"]),
                str(record["steps"]),
                f"{float(record['relative_iqr']):.2f}%",
            ]
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 7.8),
        gridspec_kw={"height_ratios": (1.0, 2.2)},
    )
    for ax in axes:
        ax.axis("off")

    axes[0].set_title("Relative-IQR summary by metric", pad=12)
    summary_table = axes[0].table(
        cellText=summary_rows,
        colLabels=("Metric", "Median", "95th percentile", "Maximum"),
        cellLoc="center",
        loc="center",
        colWidths=(0.38, 0.18, 0.22, 0.18),
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)
    summary_table.scale(1.0, 1.45)
    style_table(summary_table)

    axes[1].set_title("Ten most variable configurations", pad=12)
    detail_table = axes[1].table(
        cellText=detail_rows,
        colLabels=(
            "#",
            "Metric",
            "Backend",
            "Threads",
            "Grid",
            "Max iter.",
            "Steps",
            "Relative IQR",
        ),
        cellLoc="center",
        loc="center",
        colWidths=(0.04, 0.21, 0.15, 0.09, 0.09, 0.11, 0.09, 0.13),
    )
    detail_table.auto_set_font_size(False)
    detail_table.set_fontsize(8.5)
    detail_table.scale(1.0, 1.35)
    style_table(detail_table)

    fig.suptitle("Run-to-run variability", y=0.98)
    add_figure_note(
        fig,
        "Relative IQR = (75th percentile - 25th percentile) / median. "
        "Each configuration contains five runs.",
        bottom=0.015,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    return fig


def selected_plots(raw_choices: Sequence[str]) -> set[str]:
    choices = set(raw_choices)
    return set(PLOT_CHOICES) if "all" in choices else choices


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    apply_style()
    rows = load_rows(args.input)
    plots = selected_plots(args.plots)
    outputs: list[Path] = []

    if "cpu-scaling" in plots:
        outputs.append(
            save_figure(
                cpu_scaling_figure(
                    rows,
                    "Pure dynamics compute time",
                    "CPU strong scaling: pure dynamics",
                ),
                args.output_dir,
                "cpu_scaling_pure_dynamics",
                args.dpi,
            )
        )
        outputs.append(
            save_figure(
                cpu_scaling_figure(
                    rows,
                    "Total measured wall time",
                    "CPU strong scaling: measured solver",
                ),
                args.output_dir,
                "cpu_scaling_total_time",
                args.dpi,
            )
        )
    if "cpu-efficiency" in plots:
        outputs.append(
            save_figure(
                cpu_efficiency_figure(rows),
                args.output_dir,
                "cpu_parallel_efficiency",
                args.dpi,
            )
        )
    if "backend" in plots:
        outputs.append(
            save_figure(
                backend_comparison_figure(rows),
                args.output_dir,
                "backend_comparison",
                args.dpi,
            )
        )
    if "phases" in plots:
        outputs.append(
            save_figure(
                phase_breakdown_figure(rows),
                args.output_dir,
                "baseline_phase_times",
                args.dpi,
            )
        )
    if "grid" in plots:
        outputs.append(
            save_figure(
                grid_sweep_figure(rows),
                args.output_dir,
                "sweep_grid",
                args.dpi,
            )
        )
    if "grid-speedup" in plots:
        outputs.append(
            save_figure(
                sweep_speedup_figure(
                    rows,
                    "grid",
                    "Grid side length N",
                    "Total-time speedup across grid sizes",
                    "Square grids; 200 time steps; 4000 max fractal "
                    "iterations. CPU parallel backends use 32 threads. "
                    "Median; n=5.",
                ),
                args.output_dir,
                "grid_total_speedup",
                args.dpi,
            )
        )
    if "iterations" in plots:
        outputs.append(
            save_figure(
                iterations_sweep_figure(rows),
                args.output_dir,
                "sweep_iterations",
                args.dpi,
            )
        )
    if "iterations-speedup" in plots:
        outputs.append(
            save_figure(
                sweep_speedup_figure(
                    rows,
                    "iterations",
                    "Maximum fractal iterations",
                    "Total-time speedup across fractal-iteration limits",
                    "Grid 4000 x 4000; 200 time steps. CPU parallel "
                    "backends use 32 threads. Median; n=5.",
                ),
                args.output_dir,
                "fractal_iterations_total_speedup",
                args.dpi,
            )
        )
    if "steps" in plots:
        outputs.append(
            save_figure(
                steps_sweep_figure(rows),
                args.output_dir,
                "sweep_steps",
                args.dpi,
            )
        )
    if "steps-speedup" in plots:
        outputs.append(
            save_figure(
                sweep_speedup_figure(
                    rows,
                    "steps",
                    "Time steps",
                    "Total-time speedup across time-step counts",
                    "Grid 4000 x 4000; 4000 max fractal iterations. CPU "
                    "parallel backends use 32 threads. Median; n=5.",
                ),
                args.output_dir,
                "time_steps_total_speedup",
                args.dpi,
            )
        )
    if "variability" in plots:
        outputs.append(
            save_figure(
                variability_figure(rows),
                args.output_dir,
                "variability_summary",
                args.dpi,
            )
        )

    print(f"Generated {len(outputs)} PNG figures from {len(rows)} CSV rows")


if __name__ == "__main__":
    main()
