"""
utils/csv_writer.py

Cac ham doc/ghi file thuan tuy (CSV, JSON). Khong biet gi ve scenario, khong
biet gi ve cach chay chuong trinh. Dung chung cho engine/statistics.py va
engine/system_info.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_csv_records(path: Path) -> list[dict[str, str]]:
    """Doc toan bo dong CSV thanh list[dict]. Tra ve [] neu file khong ton tai/rong."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv_records(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
