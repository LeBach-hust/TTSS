"""
scenarios/kb5.py

Kich ban 5 (N1-10): thu tu vong lap (ijk, ikj, kij), block size, co va khong
omp simd; n = 512, 1024, 2048; ap dung cho ca matmul_seq va matmul_omp.

Da kiem tra truc tiep source:
  - matmul_seq.cpp CHI co MOT bien the tile (khong tach tile_1d/tile_2d nhu
    matmul_omp.cpp) - "tile_2d" la khai niem OpenMP collapse(2), khong ap
    dung cho ban tuan tu, nen KB5 phan tile cho seq chi dung "tile"/"tile_nosimd".
  - matmul_omp.cpp dung "tile" (=tile_1d, co omp simd) va "tile_nosimd" de
    so sanh co/khong simd tren cung mot cach phan hoach (1D).
"""

from __future__ import annotations

from utils.command_builder import make_config, RunConfig

NOTE: str | None = None

MATRIX_SIZES = (512, 1024, 2048)
BLOCK_SIZES = (16, 32, 64, 128)
LOOP_ORDERS = ("ijk", "ikj", "kij")
DEFAULT_BLOCK_FOR_LOOP_ORDER_TEST = 16


def build() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for n in MATRIX_SIZES:
        # --- Thu tu vong lap (khong tile) ---
        for variant in LOOP_ORDERS:
            configs.append(make_config(
                f"SEQ-{variant}-n{n}", "matmul_seq",
                ["-n", str(n), "-v", variant, "-b", str(DEFAULT_BLOCK_FOR_LOOP_ORDER_TEST)],
                mode="local",
            ))
            configs.append(make_config(
                f"OMP-{variant}-n{n}", "matmul_omp",
                ["-n", str(n), "-t", "4", "-v", variant, "-b", str(DEFAULT_BLOCK_FOR_LOOP_ORDER_TEST)],
                mode="local", threads=4,
            ))

        # --- Block size x co/khong omp simd ---
        for block in BLOCK_SIZES:
            for variant in ("tile_nosimd", "tile"):
                configs.append(make_config(
                    f"SEQ-{variant}-b{block}-n{n}", "matmul_seq",
                    ["-n", str(n), "-v", variant, "-b", str(block)],
                    mode="local",
                ))
                configs.append(make_config(
                    f"OMP-{variant}-b{block}-n{n}", "matmul_omp",
                    ["-n", str(n), "-t", "4", "-v", variant, "-b", str(block)],
                    mode="local", threads=4,
                ))
    return configs
