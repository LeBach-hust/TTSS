"""
scenarios/kb2.py

Kich ban 2 (N1-09): OpenMP 1, 2, 4, 8 luong; n = 512, 1024, 2048 (du 3 co du
lieu); ban chuan, tile 1D va tile 2D.

Luu y: matmul_omp.cpp co day du 3 bien the "ikj" (ban chuan), "tile"/"tile_1d"
va "tile_2d" (collapse(2)) - da kiem tra truc tiep trong source, khong suy
doan. block size co dinh = 64 (KB2 khao sat so luong, khong khao sat block -
do la viec cua KB5).
"""

from __future__ import annotations

from utils.command_builder import make_config, RunConfig

NOTE: str | None = None

MATRIX_SIZES = (512, 1024, 2048)
THREAD_COUNTS = (1, 2, 4, 8)
BLOCK_SIZE = 64

# (nhan hien thi, variant trong matmul_omp)
VARIANTS = (
    ("standard", "ikj"),
    ("tile1d", "tile_1d"),
    ("tile2d", "tile_2d"),
)


def build() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for n in MATRIX_SIZES:
        for threads in THREAD_COUNTS:
            for tag, variant in VARIANTS:
                configs.append(make_config(
                    f"OMP-{threads}-{tag}-n{n}", "matmul_omp",
                    ["-n", str(n), "-t", str(threads), "-v", variant, "-b", str(BLOCK_SIZE)],
                    mode="local", threads=threads,
                ))
    return configs
