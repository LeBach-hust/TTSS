#!/usr/bin/env python3
"""
Run project benchmarks.

Usage (khong doi so voi ban cu, de khong lam gay workflow cua ca nhom):
    python bench/run_all.py kb1
    python bench/run_all.py kb2
    python bench/run_all.py kb3
    python bench/run_all.py kb4
    python bench/run_all.py kb5
    python bench/run_all.py dot
    python bench/run_all.py all

Kien truc (xem them bench/engine, bench/scenarios, bench/utils):
    scenarios/  - CHI dinh nghia cau hinh thi nghiem (du lieu thuan tuy)
    engine/     - thuc thi (executor), kiem tra (validator), thong ke
                  (statistics), thong tin may (system_info), dieu phoi (runner)
    utils/      - ham thuan tuy dung chung (command_builder, csv_writer)
run_all.py chi la lop vo CLI mong, khong chua logic nghiep vu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phep "python bench/run_all.py ..." chay duoc tu bat ky thu muc nao,
# va cho phep cac module con import lan nhau bang "from engine import ..."
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import runner, system_info
from scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
RESULTS = ROOT / "results"

DEFAULT_REPEATS = 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parallel-computing project benchmarks.")
    parser.add_argument(
        "scenario",
        choices=(*SCENARIOS.keys(), "all"),
        help="Benchmark scenario to run.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"Measurement repetitions (default: {DEFAULT_REPEATS}).")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Per-run timeout in seconds; 0 means no timeout.")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    stamp = system_info.timestamp()
    timeout = args.timeout or None

    print("=== Parallel Computing Benchmark Runner ===")
    print(f"Scenario(s): {', '.join(names)}")
    print(f"Warm-up: 1 | Measurements/configuration: {args.repeats}")
    print(f"MPI launcher: {system_info.mpi_launcher() or 'NOT FOUND'}")

    overall_failures = 0
    for name in names:
        module = SCENARIOS[name]
        configs = module.build()
        note = getattr(module, "NOTE", None)

        report = runner.run_scenario(
            name, configs,
            root=ROOT, bin_dir=BIN, results_dir=RESULTS,
            repeats=args.repeats, timeout=timeout, note=note, stamp=stamp,
        )
        overall_failures += report.failed_or_skipped

    print(f"\nFinished. Failed/skipped configurations: {overall_failures}")
    if overall_failures:
        print("Luu y: cac cau hinh bi skip/fail khong co so do; kiem tra MPI "
              "runtime va binary truoc khi ket luan. (NOT_IMPLEMENTED khong "
              "tinh vao con so nay - xem phan [NOT_IMPLEMENTED] trong log.)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
