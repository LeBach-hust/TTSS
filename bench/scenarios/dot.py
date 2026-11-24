"""
scenarios/dot.py

Bai chung: tich vo huong 4 ban (seq, omp, mpi, hybrid), 3 co N (N3-11).
Scenario file CHI dinh nghia du lieu thi nghiem, khong thuc thi gi ca.
"""

from __future__ import annotations

from utils.command_builder import make_config, RunConfig

NOTE: str | None = None

DOT_SIZES = (100_003, 500_003, 1_000_003)


def build() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for n in DOT_SIZES:
        configs.append(make_config(
            f"dot-seq-n{n}", "dot_seq", ["-n", str(n)],
            mode="local",
        ))
        configs.append(make_config(
            f"dot-omp-4-n{n}", "dot_omp", ["-n", str(n), "-t", "4"],
            mode="local", threads=4,
        ))
        configs.append(make_config(
            f"dot-mpi-4-n{n}", "dot_mpi", ["-n", str(n)],
            mode="mpi", processes=4,
        ))
        configs.append(make_config(
            f"dot-hybrid-4x2-n{n}", "dot_hybrid", ["-n", str(n)],
            mode="hybrid", processes=4, threads=2,
        ))
    return configs
