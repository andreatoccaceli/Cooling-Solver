#!/usr/bin/env python3
"""Collect non-HDF5 benchmark results from logs/*.out into a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICES = ("acc_cpu", "acc_gpu", "cpp", "omp")
FIELDS = (
    "Grid",
    "Max fractal iterations",
    "Time steps",
    "Weight field time",
    "Weight range reduction time",
    "Initialization time",
    "Pure dynamics compute time",
    "Statistics time",
    "CSV write time",
    "Dynamics loop wall time",
    "Total measured wall time",
    "Pure dynamics performance",
    "Loop end-to-end performance",
)
HDF5_DISABLED = re.compile(r"^HDF5 compiled:\s+no\s*$", re.MULTILINE)
VALUE = re.compile(r"^([^:\n]+):\s*(.*?)\s*$", re.MULTILINE)
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_log(path: Path) -> dict[str, str] | None:
    """Return one CSV row, or None if the log is not a complete benchmark."""
    device = next((name for name in DEVICES if name in path.stem), None)
    if device is None:
        return None

    content = path.read_text(encoding="utf-8", errors="replace")
    if not HDF5_DISABLED.search(content):
        return None

    values = {key.strip(): value.strip() for key, value in VALUE.findall(content)}
    missing = [field for field in FIELDS if field not in values]
    if missing:
        print(f"Skipping {path}: missing {', '.join(missing)}", file=sys.stderr)
        return None

    row = {"Device": device, "Grid": values["Grid"]}
    for field in FIELDS[1:]:
        value = values[field].split()[0] if values[field] else ""
        if not NUMBER.fullmatch(value):
            print(f"Skipping {path}: invalid {field}: {values[field]!r}", file=sys.stderr)
            return None
        row[field] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir", type=Path, default=PROJECT_ROOT / "logs",
        help="directory containing .out logs (default: project logs directory)",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "logs" / "benchmark.csv",
        help="output CSV path (default: logs/benchmark.csv)",
    )
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        parser.error(f"logs directory does not exist: {args.logs_dir}")

    rows = []
    for path in sorted(args.logs_dir.glob("*.out")):
        row = parse_log(path)
        if row is not None:
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Device", *FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
