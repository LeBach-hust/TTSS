"""
engine/runner.py

Dieu phoi: voi mot danh sach RunConfig (tu 1 scenario), chay warm-up + N lan
do cho tung cau hinh, roi tinh summary va ghi cac file ket qua. KHONG tu
dinh nghia cau hinh (do la viec cua scenarios/*.py), KHONG tu goi subprocess
truc tiep (do la viec cua engine/executor.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.command_builder import RunConfig
from utils.csv_writer import write_csv_records, write_json, ensure_dir
from engine import executor, system_info, statistics


@dataclass
class ScenarioReport:
    name: str
    raw_csv: Path
    summary_csv: Path
    machine_json: Path
    total_configs: int
    failed_or_skipped: int
    not_implemented: int


def run_scenario(
    name: str,
    configs: list[RunConfig],
    *,
    root: Path,
    bin_dir: Path,
    results_dir: Path,
    repeats: int,
    timeout: int | None,
    note: str | None = None,
    stamp: str,
) -> ScenarioReport:
    ensure_dir(results_dir)
    raw_csv = results_dir / f"{name}_{stamp}.csv"
    summary_csv = results_dir / f"{name}_summary_{stamp}.csv"
    machine_json = results_dir / f"{name}_machine_{stamp}.json"

    mpi_exec = system_info.mpi_launcher()

    print(f"\n\n######## {name.upper()} ({len(configs)} cau hinh) ########")
    if note:
        print(f"[NOTE] {note}")

    failed_or_skipped = 0
    not_implemented = 0

    for index, config in enumerate(configs, start=1):
        print(f"\n--- {index}/{len(configs)}: {config.label} ---")

        if not config.implemented:
            print(f"[NOT_IMPLEMENTED] {config.label}: {config.skip_reason or 'chua trien khai'}")
            not_implemented += 1
            continue

        warmup = executor.run_warmup(
            config, root=root, bin_dir=bin_dir, csv_path=raw_csv,
            mpi_exec=mpi_exec, timeout=timeout,
        )
        if not warmup.ok:
            failed_or_skipped += 1
            continue

        config_ok = True
        for run_id in range(1, repeats + 1):
            outcome = executor.run_measured(
                config, run_id, root=root, bin_dir=bin_dir, csv_path=raw_csv,
                mpi_exec=mpi_exec, timeout=timeout,
            )
            if not outcome.ok:
                config_ok = False
                break
        if not config_ok:
            failed_or_skipped += 1

    summary_rows = statistics.summarize(raw_csv)
    write_csv_records(summary_csv, statistics.SUMMARY_FIELDS, summary_rows)

    metadata = system_info.machine_info()
    metadata.update({
        "scenario": name,
        "warmup_runs": 1,
        "measurement_runs_per_configuration": repeats,
        "raw_csv": raw_csv.name,
        "summary_csv": summary_csv.name,
        "total_configurations": len(configs),
        "not_implemented": not_implemented,
        "failed_or_skipped": failed_or_skipped,
    })
    write_json(machine_json, metadata)

    print(f"\n[{name.upper()}] raw:     {raw_csv.relative_to(root)}")
    print(f"[{name.upper()}] summary: {summary_csv.relative_to(root)}")
    print(f"[{name.upper()}] machine: {machine_json.relative_to(root)}")
    print(f"[{name.upper()}] not_implemented: {not_implemented}/{len(configs)}  "
          f"failed_or_skipped: {failed_or_skipped}/{len(configs)}")

    return ScenarioReport(
        name=name, raw_csv=raw_csv, summary_csv=summary_csv, machine_json=machine_json,
        total_configs=len(configs), failed_or_skipped=failed_or_skipped,
        not_implemented=not_implemented,
    )
