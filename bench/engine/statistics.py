"""
engine/statistics.py

Gom cac dong CSV tho (do C++ tu append) theo tung cau hinh (prog, variant, n,
procs, threads, block, chunk) va tinh median / IQR / min / max / std cho
total_sec, cong them median cua comp_sec/comm_sec. Dung lai dung yeu cau cua
N3-07 trong checklist.
"""

from __future__ import annotations

import statistics as pystats
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.csv_writer import read_csv_records

KEY_FIELDS = ("prog", "variant", "n", "procs", "threads", "block", "chunk")

SUMMARY_FIELDS = list(KEY_FIELDS) + [
    "samples", "median_total_sec", "iqr_total_sec", "min_total_sec",
    "max_total_sec", "std_total_sec", "median_comp_sec", "median_comm_sec",
]


def _percentile(values: list[float], q: float) -> float:
    """Noi suy tuyen tinh, tuong duong cach tinh pho bien trong bang tinh."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(raw_csv_path: Path) -> list[dict[str, Any]]:
    records = read_csv_records(raw_csv_path)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        groups[tuple(row.get(k, "") for k in KEY_FIELDS)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        totals = [float(r["total_sec"]) for r in rows]
        comps = [float(r["comp_sec"]) for r in rows]
        comms = [float(r["comm_sec"]) for r in rows]
        q1, q3 = _percentile(totals, 0.25), _percentile(totals, 0.75)
        summary_rows.append({
            **dict(zip(KEY_FIELDS, key)),
            "samples": len(totals),
            "median_total_sec": pystats.median(totals),
            "iqr_total_sec": q3 - q1,
            "min_total_sec": min(totals),
            "max_total_sec": max(totals),
            "std_total_sec": pystats.stdev(totals) if len(totals) > 1 else 0.0,
            "median_comp_sec": pystats.median(comps),
            "median_comm_sec": pystats.median(comms),
        })
    return summary_rows
