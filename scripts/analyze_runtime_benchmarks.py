#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clingrounder.evaluation.runtime_benchmark import (
    analyze_runtime_run,
    compare_runtime_runs,
    write_runtime_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare instrumented pipeline runtime runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="Run to analyze; the first run is the baseline. Repeat for comparisons.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    named_runs = []
    for value in args.run:
        if "=" not in value:
            raise SystemExit(f"Invalid --run {value!r}; expected NAME=RUN_DIR")
        name, path = value.split("=", 1)
        named_runs.append((name.strip(), analyze_runtime_run(path.strip())))
    report = compare_runtime_runs(named_runs)
    write_runtime_benchmark(report, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "runs": report["runs"]}, indent=2))


if __name__ == "__main__":
    main()
