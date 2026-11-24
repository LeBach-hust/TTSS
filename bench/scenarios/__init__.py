"""
scenarios/__init__.py

Registry anh xa ten kich ban -> module. Them kich ban moi = them 1 file
scenarios/<ten>.py (co ham build() -> list[RunConfig] va bien NOTE) roi dang
ky vao SCENARIOS ben duoi - khong can sua run_all.py hay engine/.
"""

from __future__ import annotations

from scenarios import dot, kb1, kb2, kb3, kb4, kb5

SCENARIOS = {
    "dot": dot,
    "kb1": kb1,
    "kb2": kb2,
    "kb3": kb3,
    "kb4": kb4,
    "kb5": kb5,
}
