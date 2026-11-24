"""
engine/system_info.py

Thong tin may chay (dung cho file <scenario>_machine_<timestamp>.json) va
tim mpiexec. Khong lien quan scenario/thuc thi.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
import sys
from typing import Any


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


def mpi_launcher() -> str | None:
    requested = os.environ.get("MPIEXEC", "mpiexec")
    return shutil.which(requested)


def timestamp() -> str:
    return dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
