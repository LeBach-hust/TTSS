"""
engine/validator.py

Benchmark runner KHONG lam lai viec cua tests/test_all.py (do la noi kiem tra
dung/sai chinh thuc, tranh trung lap - DRY). Validator o day chi lam hai viec
nhe, phuc vu benchmark:
  1. exit code != 0  => coi la that bai, khong tinh vao so lieu.
  2. Neu chuong trinh tu in dong "[prog | verify] ... status=PASS/FAIL"
     (cac chuong trinh MPI/Hybrid co --verify mac dinh bat) thi doc lai de
     canh bao neu FAIL (vi checksum sai thi thoi gian do duoc cung vo nghia).

Khong tu suy dien, khong tu "sua" so lieu - chi bao cao trang thai.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VERIFY_RE = re.compile(r"\[\S+ \| verify\].*status=(PASS|FAIL)")


@dataclass
class ValidationResult:
    ok: bool
    verify_status: str | None  # "PASS" | "FAIL" | None (chuong trinh khong tu verify)
    message: str = ""


def validate(returncode: int, stdout: str) -> ValidationResult:
    if returncode != 0:
        return ValidationResult(ok=False, verify_status=None, message=f"exit code {returncode}")

    match = _VERIFY_RE.search(stdout)
    if match:
        status = match.group(1)
        if status == "FAIL":
            return ValidationResult(ok=False, verify_status="FAIL", message="verify status=FAIL")
        return ValidationResult(ok=True, verify_status="PASS", message="")

    return ValidationResult(ok=True, verify_status=None, message="")
