"""
engine/executor.py

Noi DUY NHAT trong he thong goi subprocess.run. Nhan vao mot RunConfig (tu
utils/command_builder.py) va thuc thi: 1 lan warm-up (khong ghi CSV) + N lan
do (co ghi CSV), roi tra ket qua cho engine/runner.py tong hop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import subprocess

from utils.command_builder import RunConfig, build_command
from engine.validator import ValidationResult, validate


@dataclass
class RunOutcome:
    label: str
    run_id: int
    ok: bool
    validation: ValidationResult | None
    reason: str = ""  # ly do skip (binary thieu, NOT_IMPLEMENTED, mpiexec thieu...)


def _run_once(
    command: list[str],
    env: dict[str, str],
    *,
    cwd: Path,
    timeout: int | None,
    log_prefix: str,
) -> tuple[bool, str]:
    print(f"\n[{log_prefix}] {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {log_prefix}: qua thoi gian cho phep.", flush=True)
        return False, ""
    except OSError as exc:
        print(f"[ERROR] {log_prefix}: khong the khoi chay: {exc}", flush=True)
        return False, ""

    if completed.stdout:
        print(completed.stdout.rstrip())

    result = validate(completed.returncode, completed.stdout or "")
    if not result.ok:
        print(f"[FAIL] {log_prefix}: {result.message}")
        return False, completed.stdout or ""
    return True, completed.stdout or ""


def run_warmup(
    config: RunConfig,
    *,
    root: Path,
    bin_dir: Path,
    csv_path: Path,
    mpi_exec: str | None,
    timeout: int | None,
) -> RunOutcome:
    # Warm-up KHONG duoc truyen "-o": cac chuong trinh MPI/Hybrid ghi CSV chi
    # dua vao "csv_path co rong hay khong" (khong xet run_id), neu truyen -o o
    # day thi du lieu warm-up se lan vao CSV ket qua.
    run_id = 0 if config.mode == "local" else 1
    command, env, error = build_command(
        config, run_id, csv_path, bin_dir, mpi_exec, include_csv_output=False
    )
    if error:
        print(f"[SKIP] {config.label}: {error}")
        return RunOutcome(config.label, run_id, ok=False, validation=None, reason=error)

    assert command is not None
    ok, _ = _run_once(command, env, cwd=root, timeout=timeout, log_prefix=f"{config.label} warm-up")
    return RunOutcome(config.label, run_id, ok=ok, validation=None)


def run_measured(
    config: RunConfig,
    run_id: int,
    *,
    root: Path,
    bin_dir: Path,
    csv_path: Path,
    mpi_exec: str | None,
    timeout: int | None,
) -> RunOutcome:
    command, env, error = build_command(
        config, run_id, csv_path, bin_dir, mpi_exec, include_csv_output=True
    )
    if error:
        print(f"[SKIP] {config.label}: {error}")
        return RunOutcome(config.label, run_id, ok=False, validation=None, reason=error)

    assert command is not None
    ok, stdout = _run_once(
        command, env, cwd=root, timeout=timeout,
        log_prefix=f"{config.label} run {run_id}",
    )
    result = validate(0 if ok else 1, stdout)
    return RunOutcome(config.label, run_id, ok=ok, validation=result)
