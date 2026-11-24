"""
scenarios/kb1.py

Kich ban 1 (N3-09): OpenMP x4, MPI x4, Hybrid 4x2; n = 256, 512, 1024, 2048.

So sanh cong bang: ca 3 ban deu dung bien the "khong tile cuc bo" de phep so
sanh phan anh dung chi phi song song/giao tiep, khong tron voi toi uu locality
"""

from __future__ import annotations

from utils.command_builder import make_config, RunConfig

NOTE: str | None = None

MATRIX_SIZES = (256, 512, 1024, 2048)


def build() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for n in MATRIX_SIZES:
        configs.append(make_config(
            f"OMP-4-n{n}", "matmul_omp",
            ["-n", str(n), "-t", "4", "-v", "ijk"],
            mode="local", threads=4,
        ))
        configs.append(make_config(
            f"MPI-4-n{n}", "cannon_mpi",
            ["-n", str(n), "-b", "0"],
            mode="mpi", processes=4,
        ))
        configs.append(make_config(
            f"Hybrid-4x2-n{n}", "cannon_hybrid",
            ["-n", str(n)],
            mode="hybrid", processes=4, threads=2,
        ))
    return configs
