#!/usr/bin/env python3
"""Benchmark `import crewai` cold start time.

Usage:
    python scripts/benchmark_import_time.py [--runs N] [--json]

Spawns a fresh Python subprocess for each run to ensure cold imports.
Prints median, mean, min, max across all runs.
With --json, outputs machine-readable results for CI.
"""
import argparse
import json
import statistics
import subprocess
import sys


IMPORT_SCRIPT = "import time; t0 = time.perf_counter(); import crewai; print(time.perf_counter() - t0)"


def measure_import(python: str = sys.executable) -> float:
    """Run a single cold-import measurement in a subprocess."""
    result = subprocess.run(
        [python, "-c", IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        env={"PATH": "", "VIRTUAL_ENV": "", "PYTHONPATH": ""},
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Import failed: {result.stderr.strip()}")
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Benchmark crewai import time")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output JSON for CI")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fail if median exceeds this value (seconds)")
    args = parser.parse_args()

    times = []
    for i in range(args.runs):
        t = measure_import()
        times.append(t)
        if not args.json:
            print(f"  Run {i + 1}: {t:.3f}s")

    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0

    result = {
        "runs": args.runs,
        "median_s": round(median, 3),
        "mean_s": round(mean, 3),
        "stdev_s": round(stdev, 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"\n  Median: {median:.3f}s")
        print(f"  Mean:   {mean:.3f}s ± {stdev:.3f}s")
        print(f"  Range:  {min(times):.3f}s – {max(times):.3f}s")

    if args.threshold and median > args.threshold:
        print(f"\n  ❌ FAILED: median {median:.3f}s exceeds threshold {args.threshold:.3f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
