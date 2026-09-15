#!/usr/bin/env python3
"""Aggregate pytest-split duration artifacts from repeated profiling runs.

Download one or more ``test-duration-profile-*`` artifacts from GitHub Actions,
then pass either the downloaded directories or individual JSON files to this
script. It writes one CSV row per test with the observed sample count, mean,
median, p95, and maximum runtime in seconds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
import csv
import json
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Any


CSV_FIELDS = (
    "suite",
    "nodeid",
    "samples",
    "mean_seconds",
    "median_seconds",
    "p95_seconds",
    "max_seconds",
)


def duration_files(paths: Iterable[Path]) -> list[Path]:
    """Return the duration JSON files supplied directly or below supplied directories."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*-durations.json")))
        elif path.is_file() and path.name.endswith("-durations.json"):
            files.append(path)
        else:
            raise ValueError(f"Expected a duration JSON file or directory: {path}")
    if not files:
        raise ValueError("No *-durations.json files found")
    return files


def suite_name(path: Path) -> str:
    """Infer the suite from the artifact filename created by the profiling workflow."""
    return path.name.removesuffix("-durations.json")


def load_durations(path: Path) -> dict[str, float]:
    """Load and validate a pytest-split duration map."""
    try:
        raw: Any = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    durations: dict[str, float] = {}
    for nodeid, duration in raw.items():
        if not isinstance(nodeid, str) or not isinstance(duration, (int, float)):
            raise ValueError(f"Invalid test duration in {path}: {nodeid!r}")
        if duration < 0:
            raise ValueError(f"Negative test duration in {path}: {nodeid}")
        durations[nodeid] = float(duration)
    return durations


def percentile_95(values: Sequence[float]) -> float:
    """Return the nearest-rank p95 for a non-empty sequence."""
    ordered = sorted(values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def summarize(paths: Iterable[Path]) -> list[dict[str, str | int | float]]:
    """Summarize all durations by suite and pytest node ID."""
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in duration_files(paths):
        suite = suite_name(path)
        for nodeid, duration in load_durations(path).items():
            samples[(suite, nodeid)].append(duration)

    rows: list[dict[str, str | int | float]] = []
    for (suite, nodeid), values in samples.items():
        rows.append(
            {
                "suite": suite,
                "nodeid": nodeid,
                "samples": len(values),
                "mean_seconds": fmean(values),
                "median_seconds": median(values),
                "p95_seconds": percentile_95(values),
                "max_seconds": max(values),
            }
        )
    return sorted(
        rows,
        key=lambda row: (float(row["p95_seconds"]), float(row["max_seconds"])),
        reverse=True,
    )


def write_csv(rows: Iterable[dict[str, str | int | float]], output: Path) -> None:
    """Write summaries as a stable, spreadsheet-friendly CSV file."""
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Duration JSON files or directories containing them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test-duration-summary.csv"),
        help="Output CSV path",
    )
    arguments = parser.parse_args()

    rows = summarize(arguments.paths)
    write_csv(rows, arguments.output)
    sys.stdout.write(f"Wrote {len(rows)} test summaries to {arguments.output}\n")


if __name__ == "__main__":
    main()
