"""
utils/command_builder.py

Dinh nghia cau truc du lieu RunConfig (mo ta MOT cau hinh benchmark) va cac
ham thuan tuy (pure function) de bien mot RunConfig thanh:
  - argv  : danh sach tham so dong lenh thuc su se chay
  - env   : bien moi truong (OMP_NUM_THREADS, ...)

File nay KHONG goi subprocess, KHONG in ra man hinh, KHONG biet gi ve
warm-up/repeat. Thuc thi la viec cua engine/executor.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    """Mo ta mot cau hinh benchmark can chay (du lieu thuan tuy, khong hanh vi)."""

    label: str
    program: str                       # ten file thuc thi trong bin/, vd "matmul_omp"
    args: list[str] = field(default_factory=list)   # tham so rieng (vd -v ikj -b 64)
    mode: str = "local"                # "local" | "mpi" | "hybrid"
    processes: int = 1                 # so tien trinh MPI (mode mpi/hybrid)
    threads: int = 1                   # so luong OpenMP (mode local-omp/hybrid)
    implemented: bool = True           # False => khong chay, chi bao NOT_IMPLEMENTED
    skip_reason: str | None = None     # ly do neu implemented=False


def make_config(
    label: str,
    program: str,
    args: list[str],
    *,
    mode: str = "local",
    threads: int = 1,
    processes: int = 1,
    implemented: bool = True,
    skip_reason: str | None = None,
) -> RunConfig:
    return RunConfig(
        label=label,
        program=program,
        args=list(args),
        mode=mode,
        threads=threads,
        processes=processes,
        implemented=implemented,
        skip_reason=skip_reason,
    )


def find_executable(bin_dir: Path, program: str) -> Path | None:
    """Tim bin/<program> hoac bin/<program>.exe (uu tien .exe tren Windows/MSYS2)."""
    on_windows_like = os.name == "nt" or bool(os.environ.get("MSYSTEM"))
    suffixes = (".exe", "") if on_windows_like else ("", ".exe")
    for suffix in suffixes:
        candidate = bin_dir / f"{program}{suffix}"
        if candidate.is_file():
            return candidate
    return None


# Cac chuong trinh local (khong qua mpiexec) tu kiem tra "run_id > 0" truoc khi
# ghi CSV, nen co the truyen "-o" ngay ca luc warm-up (run_id=0) ma khong so bi
# ghi nham dong warm-up vao CSV.
LOCAL_SELF_GUARDED_PROGRAMS = {"dot_seq", "dot_omp", "matmul_seq", "matmul_omp"}

# Cac chuong trinh MPI/Hybrid GHI CSV CHI DUA VAO "csv_path co rong hay khong"
# (vai chuong trinh them dieu kien "pass" khi co --verify), KHONG kiem tra run_id.
# => executor tuyet doi khong duoc truyen "-o" cho cac chuong trinh nay trong
# luot warm-up, neu khong warm-up se bi ghi lan vao CSV ket qua (dung la loi
# "warm-up bi ghi vao CSV" ma de bai nhac toi).
MPI_OR_HYBRID_UNGUARDED_PROGRAMS = {
    "dot_mpi", "dot_hybrid", "cannon_mpi", "cannon_mpi_nb", "cannon_hybrid",
}


def build_command(
    config: RunConfig,
    run_id: int,
    csv_path: Path,
    bin_dir: Path,
    mpi_exec: str | None,
    *,
    include_csv_output: bool,
) -> tuple[list[str] | None, dict[str, str], str | None]:
    """
    Tra ve (argv, env, loi). Neu loi != None thi argv la None va khong duoc chay.

    include_csv_output=False  => KHONG truyen "-o" (dung cho warm-up cua
                                  MPI/Hybrid, va co the dung chung cho local).
    include_csv_output=True   => truyen "-o <csv_path>" de chuong trinh ghi CSV.
    """
    if not config.implemented:
        return None, os.environ.copy(), config.skip_reason or "Chua trien khai (NOT_IMPLEMENTED)"

    exe = find_executable(bin_dir, config.program)
    if exe is None:
        return None, os.environ.copy(), f"Thieu binary: bin/{config.program}[.exe]"

    env = os.environ.copy()
    # Luon set OMP_NUM_THREADS: can thiet cho cannon_hybrid/dot_hybrid (khong co
    # flag -t, chi doc OMP_NUM_THREADS/omp_get_max_threads()); vo hai voi cac
    # chuong trinh da nhan -t rieng.
    env["OMP_NUM_THREADS"] = str(config.threads)

    args = list(config.args)
    args += ["-r", str(run_id)]
    if include_csv_output:
        args += ["-o", str(csv_path)]

    if config.mode in ("mpi", "hybrid"):
        if mpi_exec is None:
            return None, env, "Khong tim thay mpiexec (MPI runtime chua san sang)"
        command = [mpi_exec, "-n", str(config.processes), str(exe), *args]
    else:
        command = [str(exe), *args]

    return command, env, None
