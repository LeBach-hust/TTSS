"""
scenarios/kb3.py

Kich ban 3 (N2-09): Cannon chan (cannon_mpi) va khong chan (cannon_mpi_nb);
p = 4, 9; n = 576, 1152, 2304; tinh ti le che giau (comm_sec vs comp_sec).

QUAN TRONG: KB3 chi lien quan Cannon MPI (khong co Hybrid), nen KHONG bi anh
huong boi viec cannon_hybrid_nb.cpp chua trien khai. Da kiem tra truc tiep
cannon_mpi_nb.cpp: day la ban Isend/Irecv/Waitall day du, khong phai stub
(ban run_all.py cu ghi nham comment la stub va bo qua nhanh nay - loi do da
duoc sua o day).

Ti le che giau duoc tinh tu comm_sec/comp_sec sau khi co CSV (viec cua
plot.py / phan tich bao cao), khong tinh trong luc chay benchmark.
"""

from __future__ import annotations

from utils.command_builder import make_config, RunConfig

NOTE: str | None = None

MATRIX_SIZES = (576, 1152, 2304)
PROCESS_COUNTS = (4, 9)
BLOCK_SIZE = 0  # khong tile cuc bo: do dung ti le che giau giao tiep/tinh toan


def build() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for n in MATRIX_SIZES:
        for p in PROCESS_COUNTS:
            configs.append(make_config(
                f"MPI-blocking-p{p}-n{n}", "cannon_mpi",
                ["-n", str(n), "-b", str(BLOCK_SIZE)],
                mode="mpi", processes=p,
            ))
            configs.append(make_config(
                f"MPI-nonblocking-p{p}-n{n}", "cannon_mpi_nb",
                ["-n", str(n), "-b", str(BLOCK_SIZE)],
                mode="mpi", processes=p,
            ))
    return configs
