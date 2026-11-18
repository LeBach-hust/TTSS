#!/usr/bin/env python3
"""
Run project benchmarks.

Usage:
    python bench/run_all.py kb1
    python bench/run_all.py kb2
    python bench/run_all.py kb3
    python bench/run_all.py kb4
    python bench/run_all.py kb5
    python bench/run_all.py dot
    python bench/run_all.py all

Each configuration is executed once as warm-up and five times for measurement.
Raw records are appended by the C++ programs to one timestamped CSV.
A summary CSV and machine metadata JSON are written beside it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
RESULTS = ROOT / "results"

REPEATS = 5
MATRIX_SIZES = (256, 512, 1024, 2048)
BLOCK_SIZES = (16, 32, 64, 128)
DOT_SIZES = (100003, 500003, 1000003)

CSV_FIELDS = [
    "prog", "variant", "n", "procs", "threads", "block", "chunk", "run",
    "init_sec", "comp_sec", "comm_sec", "total_sec", "checksum",
]


def timestamp() -> str:
    return dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def machine_info() -> dict[str, Any]:
    return {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.uname().processor,
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "compiler_environment": {
            key: os.environ.get(key)
            for key in ("CXX", "CC", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
        },
        "mpi_launcher": os.environ.get("MPIEXEC", "mpiexec"),
    }


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, equivalent to common spreadsheet methods."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def executable(name: str) -> Path | None:
    suffixes = (".exe", "") if os.name == "nt" or os.environ.get("MSYSTEM") else ("", ".exe")
    for suffix in suffixes:
        candidate = BIN / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def mpi_launcher() -> str | None:
    requested = os.environ.get("MPIEXEC", "mpiexec")
    return shutil.which(requested)


def run_command(
    label: str,
    command: list[str],
    env: dict[str, str],
    *,
    timeout: int | None = None,
) -> bool:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {label}: quá thời gian cho phép.", flush=True)
        return False
    except OSError as exc:
        print(f"[ERROR] {label}: không thể khởi chạy: {exc}", flush=True)
        return False

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        print(f"[FAIL] {label}: exit code {completed.returncode}")
        return False
    return True


def make_config(
    label: str,
    program: str,
    args: list[str],
    *,
    mode: str,
    threads: int = 1,
    processes: int = 1,
) -> dict[str, Any]:
    return {
        "label": label,
        "program": program,
        "args": args,
        "mode": mode,  # local | mpi | hybrid
        "threads": threads,
        "processes": processes,
    }


def build_scenarios() -> dict[str, list[dict[str, Any]]]:
    scenarios: dict[str, list[dict[str, Any]]] = {
        "kb1": [],
        "kb2": [],
        "kb3": [],
        "kb4": [],
        "kb5": [],
        "dot": [],
    }

    # KB1: OpenMP x4, MPI x4, Hybrid 4x2; N = 256, 512, 1024, 2048.
    for n in MATRIX_SIZES:
        scenarios["kb1"].append(make_config(
            f"OMP-4-n{n}", "matmul_omp",
            ["-n", str(n), "-t", "4", "-v", "ijk", "-b", "16"],
            mode="local", threads=4,
        ))
        scenarios["kb1"].append(make_config(
            f"MPI-4-n{n}", "cannon_mpi",
            ["-n", str(n), "-b", "0"],
            mode="mpi", processes=4,
        ))
        scenarios["kb1"].append(make_config(
            f"Hybrid-4x2-n{n}", "cannon_hybrid",
            ["-n", str(n)],
            mode="hybrid", processes=4, threads=2,
        ))

    # KB2: OpenMP 1/2/4/8 threads, standard and tiled, N=1024/2048.
    for n in (1024, 2048):
        for threads in (1, 2, 4, 8):
            for variant, block in (("ijk", 16), ("tile", 16)):
                scenarios["kb2"].append(make_config(
                    f"OMP-{threads}-{variant}-n{n}", "matmul_omp",
                    ["-n", str(n), "-t", str(threads), "-v", variant, "-b", str(block)],
                    mode="local", threads=threads,
                ))

    # KB3: blocking/nonblocking Cannon, p=4/9, N=576/1152/2304.
    # Nonblocking source is currently a TODO stub, so only blocking is runnable.
    for n in (576, 1152, 2304):
        for processes in (4, 9):
            scenarios["kb3"].append(make_config(
                f"MPI-blocking-p{processes}-n{n}", "cannon_mpi",
                ["-n", str(n), "-b", "0"],
                mode="mpi", processes=processes,
            ))

    # KB4 requires the hybrid nonblocking implementation and configurable
    # MPI_Testall polling. That source is currently a TODO stub.
    # Keep the scenario explicit but do not run a nonfunctional binary.

    # KB5: loop order, block size, with/without omp simd.
    for n in (1024, 2048):
        for variant in ("ijk", "ikj", "kij"):
            scenarios["kb5"].append(make_config(
                f"SEQ-{variant}-n{n}", "matmul_seq",
                ["-n", str(n), "-v", variant, "-b", "16"],
                mode="local",
            ))
            scenarios["kb5"].append(make_config(
                f"OMP-{variant}-n{n}", "matmul_omp",
                ["-n", str(n), "-t", "4", "-v", variant, "-b", "16"],
                mode="local", threads=4,
            ))
        for block in BLOCK_SIZES:
            for variant in ("tile_nosimd", "tile"):
                scenarios["kb5"].append(make_config(
                    f"SEQ-{variant}-b{block}-n{n}", "matmul_seq",
                    ["-n", str(n), "-v", variant, "-b", str(block)],
                    mode="local",
                ))
                scenarios["kb5"].append(make_config(
                    f"OMP-{variant}-b{block}-n{n}", "matmul_omp",
                    ["-n", str(n), "-t", "4", "-v", variant, "-b", str(block)],
                    mode="local", threads=4,
                ))

    # Common dot benchmark: sequential, OpenMP, MPI, Hybrid.
    for n in DOT_SIZES:
        scenarios["dot"].append(make_config(
            f"dot-seq-n{n}", "dot_seq", ["-n", str(n)],
            mode="local",
        ))
        scenarios["dot"].append(make_config(
            f"dot-omp-4-n{n}", "dot_omp", ["-n", str(n), "-t", "4"],
            mode="local", threads=4,
        ))
        scenarios["dot"].append(make_config(
            f"dot-mpi-4-n{n}", "dot_mpi", ["-n", str(n)],
            mode="mpi", processes=4,
        ))
        scenarios["dot"].append(make_config(
            f"dot-hybrid-4x2-n{n}", "dot_hybrid", ["-n", str(n)],
            mode="hybrid", processes=4, threads=2,
        ))

    return scenarios


def command_for(
    config: dict[str, Any],
    run_id: int,
    csv_path: Path,
    mpi_exec: str | None,
) -> tuple[list[str] | None, dict[str, str], str | None]:
    program = config["program"]
    exe = executable(program)
    if exe is None:
        return None, os.environ.copy(), f"Thiếu binary: bin/{program}[.exe]"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config["threads"])

    args = list(config["args"])
    # The local dot and matmul programs require -o even for warm-up.
    # Their C++ code suppresses CSV output when run_id == 0.
    if program in ("dot_seq", "dot_omp", "matmul_seq", "matmul_omp"):
        args += ["-r", str(run_id), "-o", str(csv_path)]
    else:
        args += ["-r", str(run_id)]
        # MPI/Hybrid implementations append whenever -o is non-empty.
        # Omitting -o for warm-up prevents warm-up records from polluting CSV.
        if run_id > 0:
            args += ["-o", str(csv_path)]

    if config["mode"] == "mpi":
        if mpi_exec is None:
            return None, env, "Không tìm thấy mpiexec (MPI runtime chưa sẵn sàng)"
        command = [mpi_exec, "-n", str(config["processes"]), str(exe), *args]
    elif config["mode"] == "hybrid":
        if mpi_exec is None:
            return None, env, "Không tìm thấy mpiexec (MPI runtime chưa sẵn sàng)"
        command = [mpi_exec, "-n", str(config["processes"]), str(exe), *args]
    else:
        command = [str(exe), *args]

    return command, env, None


def read_records(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_summary(raw_path: Path, summary_path: Path) -> None:
    records = read_records(raw_path)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    key_fields = ("prog", "variant", "n", "procs", "threads", "block", "chunk")
    for row in records:
        groups[tuple(row.get(k, "") for k in key_fields)].append(row)

    fields = list(key_fields) + [
        "samples", "median_total_sec", "iqr_total_sec", "min_total_sec",
        "max_total_sec", "std_total_sec", "median_comp_sec", "median_comm_sec",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            totals = [float(r["total_sec"]) for r in rows]
            comps = [float(r["comp_sec"]) for r in rows]
            comms = [float(r["comm_sec"]) for r in rows]
            q1, q3 = percentile(totals, 0.25), percentile(totals, 0.75)
            writer.writerow({
                **dict(zip(key_fields, key)),
                "samples": len(totals),
                "median_total_sec": statistics.median(totals),
                "iqr_total_sec": q3 - q1,
                "min_total_sec": min(totals),
                "max_total_sec": max(totals),
                "std_total_sec": statistics.stdev(totals) if len(totals) > 1 else 0.0,
                "median_comp_sec": statistics.median(comps),
                "median_comm_sec": statistics.median(comms),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parallel-computing project benchmarks.")
    parser.add_argument(
        "scenario",
        choices=("kb1", "kb2", "kb3", "kb4", "kb5", "dot", "all"),
        help="Benchmark scenario to run.",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS, help="Measurement repetitions (default: 5).")
    parser.add_argument("--timeout", type=int, default=0, help="Per-run timeout in seconds; 0 means no timeout.")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    selected = build_scenarios()
    names = list(selected) if args.scenario == "all" else [args.scenario]
    mpi_exec = mpi_launcher()

    print("=== Parallel Computing Benchmark Runner ===")
    print(f"Scenario(s): {', '.join(names)}")
    print(f"Warm-up: 1 | Measurements/configuration: {args.repeats}")
    print(f"MPI launcher: {mpi_exec or 'NOT FOUND'}")

    overall_failures = 0
    for scenario in names:
        raw_path = RESULTS / f"{scenario}_{stamp}.csv"
        summary_path = RESULTS / f"{scenario}_summary_{stamp}.csv"
        metadata_path = RESULTS / f"{scenario}_machine_{stamp}.json"

        metadata = machine_info()
        metadata.update({
            "scenario": scenario,
            "warmup_runs": 1,
            "measurement_runs_per_configuration": args.repeats,
            "raw_csv": raw_path.name,
            "summary_csv": summary_path.name,
        })
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        configs = selected[scenario]
        if scenario == "kb4":
            print("\n[SKIP] KB4: cannon_hybrid_nb.cpp hiện là TODO stub; chưa có cấu hình chạy hợp lệ.")
            overall_failures += 1
            continue

        print(f"\n\n######## {scenario.upper()} ({len(configs)} configurations) ########")
        scenario_failures = 0
        if scenario == "kb3":
            print("[PARTIAL] Chỉ chạy Cannon blocking. cannon_mpi_nb.cpp hiện là TODO stub,")
            print("          nên nhánh nonblocking của KB3 chưa thể benchmark.")
            scenario_failures += 1
        for index, config in enumerate(configs, start=1):
            print(f"\n--- {index}/{len(configs)}: {config['label']} ---")
            # Run once as warm-up. Local programs accept run=0 and suppress CSV.
            # MPI/Hybrid warm-up uses run=1 but omits -o, so it cannot append CSV.
            warmup_id = 0 if config["mode"] == "local" else 1
            command, env, reason = command_for(config, warmup_id, raw_path, mpi_exec)
            if reason:
                print(f"[SKIP] {config['label']}: {reason}")
                scenario_failures += 1
                continue
            assert command is not None
            if not run_command(
                f"{config['label']} warm-up", command, env,
                timeout=args.timeout or None,
            ):
                scenario_failures += 1
                continue

            config_ok = True
            for run_id in range(1, args.repeats + 1):
                command, env, reason = command_for(config, run_id, raw_path, mpi_exec)
                if reason:
                    print(f"[SKIP] {config['label']}: {reason}")
                    config_ok = False
                    break
                assert command is not None
                if not run_command(
                    f"{config['label']} run {run_id}/{args.repeats}",
                    command, env, timeout=args.timeout or None,
                ):
                    config_ok = False
                    break
            if not config_ok:
                scenario_failures += 1

        write_summary(raw_path, summary_path)
        overall_failures += scenario_failures
        print(f"\n[{scenario.upper()}] raw: {raw_path.relative_to(ROOT)}")
        print(f"[{scenario.upper()}] summary: {summary_path.relative_to(ROOT)}")
        print(f"[{scenario.upper()}] machine: {metadata_path.relative_to(ROOT)}")
        print(f"[{scenario.upper()}] failed/skipped configurations: {scenario_failures}/{len(configs)}")

    print(f"\nFinished. Failed/skipped configurations: {overall_failures}")
    if overall_failures:
        print("Lưu ý: các cấu hình bị skip/fail không có số đo; kiểm tra MPI runtime và binary trước khi kết luận.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
