#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "fig"


def read_csv(path: Path) -> list[dict]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        print(f"[WARN] Cannot read {path.name}: {exc}")
        return []


def to_float(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def to_int(value, default=None):
    number = to_float(value, default)
    return int(number) if number is not None else default


def latest_nonempty_csv(prefix: str) -> tuple[Path | None, list[dict]]:
    """Pick the newest non-summary CSV for a benchmark family."""
    candidates = sorted(
        p for p in RESULTS_DIR.glob(f"{prefix}*.csv")
        if "summary" not in p.stem.lower()
    )
    usable = []
    for path in candidates:
        rows = [r for r in read_csv(path) if any(str(v).strip() for v in r.values())]
        
        if rows and "total_sec" not in rows[0]:
            continue
        if rows:
            usable.append((path, rows))
    if not usable:
        return None, []
    # Timestamped filenames sort chronologically; the last nonempty file wins.
    return usable[-1]


def load_family(prefix: str) -> tuple[list[dict], Path | None]:
    path, rows = latest_nonempty_csv(prefix)
    normalized = []
    for row in rows:
        item = dict(row)
        item["_source"] = prefix.upper()
        item["_file"] = path.name if path else ""
        item["_n"] = to_int(row.get("n") or row.get("size"))
        item["_procs"] = to_int(row.get("procs") or row.get("processes"), 1)
        item["_threads"] = to_int(row.get("threads"), 1)
        item["_comp"] = to_float(row.get("comp_sec"))
        item["_comm"] = to_float(row.get("comm_sec"), 0.0)
        item["_total"] = to_float(row.get("total_sec"))
        item["_variant"] = (row.get("variant") or "").strip()
        item["_prog"] = (row.get("prog") or "").strip()
        item["_scenario"] = (row.get("scenario") or "").strip()
        normalized.append(item)
    return normalized, path


def median(values: list[float]) -> float:
    return statistics.median(values)


def save_figure(filename: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path.relative_to(ROOT)}")


def plot_stacked_timing(rows: list[dict], source: str):
    valid = [
        r for r in rows
        if r["_comp"] is not None and r["_comm"] is not None and r["_n"] is not None
    ]
    if not valid:
        print(f"[SKIP] {source}: no comp_sec/comm_sec data for stacked chart")
        return

    groups = defaultdict(lambda: {"comp": [], "comm": []})
    for r in valid:
        label = r["_prog"] or r["_scenario"] or r["_variant"] or source
        if r["_prog"]:
            label = f'{label}/{r["_variant"]}'
        label += f' | n={r["_n"]}, p={r["_procs"]}, t={r["_threads"]}'
        if r.get("chunk"):
            label += f', chunk={r["chunk"]}'
        groups[label]["comp"].append(r["_comp"])
        groups[label]["comm"].append(r["_comm"])

    labels = list(groups)
    comp = [median(groups[k]["comp"]) for k in labels]
    comm = [median(groups[k]["comm"]) for k in labels]

    width = max(9, min(18, len(labels) * 0.75))
    plt.figure(figsize=(width, 6))
    x = list(range(len(labels)))
    plt.bar(x, comp, label="Tính toán (comp_sec)")
    plt.bar(x, comm, bottom=comp, label="Truyền thông (comm_sec)")
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("Thời gian (giây)")
    plt.title(f"{source}: thời gian tính toán và truyền thông")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure(f"{source.lower()}_timing_comp_comm.png")


def find_seq_baselines(all_rows: list[dict]) -> dict[int, float]:
    """Find sequential ijk baselines, preferably from KB5."""
    samples = defaultdict(list)

    for r in all_rows:
        n = r["_n"]
        total = r["_total"]
        if n is None or total is None or total <= 0:
            continue

        prog = r["_prog"].lower()
        variant = r["_variant"].lower()
        scenario = r["_scenario"].lower()

        is_seq = (
            prog == "matmul_seq"
            or scenario.startswith("seq-")
            or scenario.startswith("seq_")
        )
        is_ijk = (
            variant == "ijk"
            or "seq-ijk" in scenario
            or "seq_ijk" in scenario
            or scenario.startswith("seq-ijk-")
        )

        if is_seq and is_ijk:
            samples[n].append(total)

    return {n: median(values) for n, values in samples.items()}


def plot_speedup_efficiency(kb1_rows: list[dict], baselines: dict[int, float]):
    if not kb1_rows:
        print("[SKIP] Speedup/efficiency: KB1 raw data is unavailable")
        return
    if not baselines:
        print(
            "[SKIP] Speedup/efficiency: no sequential matmul_seq/SEQ-ijk "
            "baseline found. KB1 itself contains only parallel configurations."
        )
        return

    groups = defaultdict(list)
    for r in kb1_rows:
        n, total = r["_n"], r["_total"]
        if n not in baselines or total is None or total <= 0:
            continue
        p_eff = (r["_procs"] or 1) * (r["_threads"] or 1)
        label = r["_prog"] or r["_variant"] or "parallel"
        if r["_prog"] == "matmul_omp":
            label = f'OpenMP ({p_eff} threads)'
        elif "hybrid" in r["_prog"]:
            label = f'Hybrid ({r["_procs"]} MPI x {r["_threads"]} threads)'
        elif "mpi" in r["_prog"]:
            label = f'MPI ({r["_procs"]} processes)'
        groups[(label, n, p_eff)].append(total)

    if not groups:
        print("[SKIP] Speedup/efficiency: no common n between KB1 and sequential baseline")
        return

    series = defaultdict(list)
    for (label, n, p_eff), times in groups.items():
        parallel_time = median(times)
        speedup = baselines[n] / parallel_time
        efficiency = speedup / p_eff
        series[label].append((n, speedup, efficiency))

    plt.figure(figsize=(9, 6))
    for label, points in sorted(series.items()):
        points.sort()
        plt.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=label,
        )
    plt.xlabel("Kích thước ma trận n")
    plt.ylabel("Speedup")
    plt.title("KB1: Speedup theo kích thước ma trận")
    plt.xticks(sorted({p[0] for values in series.values() for p in values}))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_figure("kb1_speedup_by_n.png")

    plt.figure(figsize=(9, 6))
    for label, points in sorted(series.items()):
        points.sort()
        plt.plot(
            [p[0] for p in points],
            [p[2] for p in points],
            marker="o",
            label=label,
        )
    plt.xlabel("Kích thước ma trận n")
    plt.ylabel("Efficiency")
    plt.title("KB1: Efficiency theo kích thước ma trận")
    plt.xticks(sorted({p[0] for values in series.values() for p in values}))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_figure("kb1_efficiency_by_n.png")


def extract_loop_order(row: dict) -> str | None:
    value = (row.get("loop_order") or "").strip().lower()
    if value in {"ijk", "ikj", "kij"}:
        return value
    text = (row.get("scenario") or "").lower()
    match = re.search(r"(?:seq|omp)-(ijk|ikj|kij)(?:-|_|$)", text)
    return match.group(1) if match else None


def extract_tile_size(row: dict) -> int | None:
    value = to_int(row.get("tile_size") or row.get("block"))
    if value is not None:
        return value
    text = (row.get("scenario") or "").lower()
    match = re.search(r"(?:^|[-_])b(\d+)(?:[-_]|$)", text)
    return int(match.group(1)) if match else None


def extract_simd(row: dict) -> str:
    value = (row.get("simd") or "").strip().lower()
    if value in {"1", "true", "yes", "on", "simd"}:
        return "SIMD"
    if value in {"0", "false", "no", "off", "nosimd"}:
        return "No SIMD"
    scenario = (row.get("scenario") or "").lower()
    variant = (row.get("variant") or "").lower()
    if "nosimd" in scenario or "nosimd" in variant:
        return "No SIMD"
    if "tile" in scenario or "tile" in variant:
        return "SIMD"
    return "Unknown"


def plot_kb4(rows: list[dict]):
    valid = [r for r in rows if r["_total"] is not None]
    if not valid:
        print("[SKIP] KB4: CSV chưa có dữ liệu")
        return

    groups = defaultdict(lambda: {"total": [], "comp": [], "comm": []})
    for r in valid:
        poll = (r.get("poll_interval") or "").strip()
        if not poll:
            continue
        groups[poll]["total"].append(r["_total"])
        if r["_comp"] is not None:
            groups[poll]["comp"].append(r["_comp"])
        if r["_comm"] is not None:
            groups[poll]["comm"].append(r["_comm"])

    order = ["none", "64", "16", "4"]
    labels = [x for x in order if x in groups]
    labels += sorted(x for x in groups if x not in labels)
    if not labels:
        print("[SKIP] KB4: không tìm thấy poll_interval hợp lệ")
        return

    plt.figure(figsize=(9, 6))
    for key, title in (("total", "Total"), ("comp", "Computation"), ("comm", "Communication")):
        points = [
            (i, median(groups[label][key]))
            for i, label in enumerate(labels)
            if groups[label][key]
        ]
        if points:
            plt.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                label=title,
            )
    plt.xticks(range(len(labels)), labels)
    plt.xlabel("Tần suất gọi MPI_Testall (số hàng/chunk)")
    plt.ylabel("Thời gian (giây)")
    plt.title("KB4: Ảnh hưởng của tần suất MPI_Testall")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("kb4_polling_frequency.png")


def plot_kb5(rows: list[dict]):
    valid = [r for r in rows if r["_total"] is not None]
    if not valid:
        print("[SKIP] KB5: CSV chưa có dữ liệu")
        return

    # Hình 1: so sánh thứ tự vòng lặp.
    loop_groups = defaultdict(list)
    for r in valid:
        order = extract_loop_order(r)
        if order and r["_n"] is not None:
            label = f'{r["_prog"] or r["_scenario"]} | {order}'
            loop_groups[(label, r["_n"])].append(r["_total"])

    if loop_groups:
        plt.figure(figsize=(10, 6))
        for (label, n), values in sorted(loop_groups.items()):
            plt.scatter(n, median(values), label=label)
        plt.xlabel("Kích thước ma trận n")
        plt.ylabel("Thời gian (giây)")
        plt.title("KB5: Ảnh hưởng của thứ tự vòng lặp")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        save_figure("kb5_loop_order.png")
    else:
        print("[SKIP] KB5 loop-order: không tìm thấy loop_order")

    # Hình 2: ảnh hưởng tile size, tách SIMD/No SIMD.
    tile_groups = defaultdict(list)
    for r in valid:
        tile = extract_tile_size(r)
        if tile is None:
            continue
        simd = extract_simd(r)
        n = r["_n"] if r["_n"] is not None else "?"
        label = f'{r["_prog"] or r["_scenario"]} | n={n} | {simd}'
        tile_groups[label, tile].append(r["_total"])

    if tile_groups:
        plt.figure(figsize=(10, 6))
        for (label, tile), values in sorted(tile_groups.items()):
            plt.scatter(tile, median(values), label=label)
        plt.xlabel("Tile size")
        plt.ylabel("Thời gian (giây)")
        plt.title("KB5: Ảnh hưởng của tile size và SIMD")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        save_figure("kb5_tile_size_simd.png")
    else:
        print("[SKIP] KB5 tile-size: không tìm thấy tile_size")

KB2_VARIANT_LABELS = {
    "ikj": "Standard (ikj)",
    "tile": "Tile 1D",
    "tile_1d": "Tile 1D",
    "tile_2d": "Tile 2D",
}
KB2_VARIANT_ORDER = ["ikj", "tile_1d", "tile_2d"]


def _kb2_groups(rows: list[dict]) -> dict[tuple[str, int, int], list[float]]:
    """Gom thoi gian total theo (variant, n, threads)."""
    groups = defaultdict(list)
    for r in rows:
        if r["_prog"] and r["_prog"] != "matmul_omp":
            continue
        if r["_n"] is None or r["_total"] is None or r["_total"] <= 0:
            continue
        variant = r["_variant"].lower() or "unknown"
        if variant == "tile":
            variant = "tile_1d"
        groups[(variant, r["_n"], r["_threads"] or 1)].append(r["_total"])
    return dict(groups)

# Kich ban 2: OpenMP 1, 2, 4, 8 luong; n = 512, 1024, 2048; ban chuan, tile 1D va tile 2D.
def plot_kb2(rows: list[dict]):
    groups = _kb2_groups(rows)
    if not groups:
        print("[SKIP] KB2: CSV chưa có dữ liệu matmul_omp")
        return

    ns = sorted({k[1] for k in groups})
    threads_all = sorted({k[2] for k in groups})
    found = {k[0] for k in groups}
    variants = [v for v in KB2_VARIANT_ORDER if v in found] + sorted(found - set(KB2_VARIANT_ORDER))

    def label(v):
        return KB2_VARIANT_LABELS.get(v, v)

    #Hình 1: thời gian theo số luồng, mỗi n một ô, có thanh min-max
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4.5), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for v in variants:
            pts = [(t, groups[(v, n, t)]) for t in threads_all if (v, n, t) in groups]
            if not pts:
                continue
            ts = [p[0] for p in pts]
            med = [median(p[1]) for p in pts]
            lo = [m - min(p[1]) for m, p in zip(med, pts)]
            hi = [max(p[1]) - m for m, p in zip(med, pts)]
            ax.errorbar(ts, med, yerr=[lo, hi], marker="o", capsize=3, label=label(v))
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Số luồng OpenMP")
        ax.set_xticks(threads_all)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Thời gian (giây)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("KB2: Thời gian chạy theo số luồng (trung vị, thanh = min–max)")
    save_figure("kb2_time_by_threads.png")

    #Tính speedup / efficiency so với chính biến thể đó ở 1 luồng
    scaling = {}  # (v, n, t) -> (speedup, efficiency)
    for (v, n, t), values in groups.items():
        base = groups.get((v, n, 1))
        if base:
            s = median(base) / median(values)
            scaling[(v, n, t)] = (s, s / t)
    if not scaling:
        print("[SKIP] KB2 speedup/efficiency: thiếu dữ liệu 1 luồng làm mốc")
    else:
        #Hình 2 và 3: speedup, efficiency; mỗi biến thể một ô
        for idx, (fname, ylabel, title) in enumerate((
            ("kb2_speedup.png", "Speedup  T(1)/T(t)", "KB2: Speedup theo số luồng"),
            ("kb2_efficiency.png", "Efficiency  S(t)/t", "KB2: Efficiency theo số luồng"),
        )):
            fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4.5),
                                     squeeze=False, sharey=True)
            for ax, v in zip(axes[0], variants):
                for n in ns:
                    pts = [(t, scaling[(v, n, t)][idx]) for t in threads_all if (v, n, t) in scaling]
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"n = {n}")
                ideal = threads_all if idx == 0 else [1.0] * len(threads_all)
                ax.plot(threads_all, ideal, "k--", alpha=0.5, label="Lý tưởng")
                ax.set_title(label(v))
                ax.set_xlabel("Số luồng OpenMP")
                ax.set_xticks(threads_all)
                ax.grid(alpha=0.3)
            axes[0][0].set_ylabel(ylabel)
            axes[0][0].legend(fontsize=8)
            fig.suptitle(title)
            save_figure(fname)

    # Hình 4: GFLOP/s, so sánh 3 biến thể; mỗi n một ô
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4.5), squeeze=False)
    width = 0.8 / len(variants)
    for ax, n in zip(axes[0], ns):
        for i, v in enumerate(variants):
            xs, ys = [], []
            for j, t in enumerate(threads_all):
                if (v, n, t) in groups:
                    xs.append(j + (i - (len(variants) - 1) / 2) * width)
                    ys.append(2 * n ** 3 / median(groups[(v, n, t)]) / 1e9)
            if xs:
                ax.bar(xs, ys, width, label=label(v))
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Số luồng OpenMP")
        ax.set_xticks(range(len(threads_all)))
        ax.set_xticklabels(threads_all)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("GFLOP/s  (2n³ / thời gian)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("KB2: Hiệu năng GFLOP/s theo biến thể và số luồng")
    save_figure("kb2_gflops.png")

    
    output = RESULTS_DIR / "kb2_results.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "n", "threads", "samples", "median_sec", "min_sec",
                         "max_sec", "speedup", "efficiency", "gflops"])
        for (v, n, t) in sorted(groups, key=lambda k: (k[1], variants.index(k[0]), k[2])):
            vals = groups[(v, n, t)]
            s, e = scaling.get((v, n, t), ("", ""))
            writer.writerow([
                v, n, t, len(vals), f"{median(vals):.6f}", f"{min(vals):.6f}",
                f"{max(vals):.6f}",
                f"{s:.3f}" if s != "" else "", f"{e:.3f}" if e != "" else "",
                f"{2 * n ** 3 / median(vals) / 1e9:.3f}",
            ])
    print(f"[OK] {output.relative_to(ROOT)}")

def write_summary(all_rows: list[dict]):
    if not all_rows:
        print("[SKIP] Summary: không có dữ liệu benchmark")
        return

    keys = (
        "_source", "_prog", "_variant", "_n", "_procs", "_threads",
        "block", "chunk", "poll_interval", "loop_order", "tile_size", "simd",
    )
    groups = defaultdict(lambda: {"comp": [], "comm": [], "total": [], "runs": 0})

    for r in all_rows:
        key = tuple(r.get(k, "") if k.startswith("_") else r.get(k, "") for k in keys)
        g = groups[key]
        g["runs"] += 1
        if r["_comp"] is not None:
            g["comp"].append(r["_comp"])
        if r["_comm"] is not None:
            g["comm"].append(r["_comm"])
        if r["_total"] is not None:
            g["total"].append(r["_total"])

    output = RESULTS_DIR / "benchmark_summary.csv"
    fieldnames = [
        "source", "prog", "variant", "n", "procs", "threads",
        "block", "chunk", "poll_interval", "loop_order", "tile_size", "simd",
        "samples", "median_comp_sec", "median_comm_sec", "median_total_sec",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key, values in sorted(groups.items(), key=lambda x: str(x[0])):
            row = dict(zip(keys, key))
            writer.writerow({
                "source": row["_source"],
                "prog": row["_prog"],
                "variant": row["_variant"],
                "n": row["_n"],
                "procs": row["_procs"],
                "threads": row["_threads"],
                "block": row["block"],
                "chunk": row["chunk"],
                "poll_interval": row["poll_interval"],
                "loop_order": row["loop_order"],
                "tile_size": row["tile_size"],
                "simd": row["simd"],
                "samples": values["runs"],
                "median_comp_sec": median(values["comp"]) if values["comp"] else "",
                "median_comm_sec": median(values["comm"]) if values["comm"] else "",
                "median_total_sec": median(values["total"]) if values["total"] else "",
            })

    print(f"[OK] {output.relative_to(ROOT)}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    family_rows = {}
    print("========== LOAD BENCHMARK CSV ==========")

    for prefix in ("kb1", "kb2", "kb3", "kb4", "kb5"):
        rows, path = load_family(prefix)
        family_rows[prefix] = rows
        if path:
            print(f"[LOAD] {path.name}: {len(rows)} rows")
            all_rows.extend(rows)
        else:
            print(f"[SKIP] {prefix.upper()}: no nonempty CSV")

    # Include the dedicated N3-06 Hybrid nonblocking test file if present.
    n3_path = RESULTS_DIR / "n3_06_hybrid_nb.csv"
    if n3_path.exists():
        n3_rows = []
        for row in read_csv(n3_path):
            item = dict(row)
            item["_source"] = "N3_06"
            item["_file"] = n3_path.name
            item["_n"] = to_int(row.get("n") or row.get("size"))
            item["_procs"] = to_int(row.get("procs") or row.get("processes"), 1)
            item["_threads"] = to_int(row.get("threads"), 1)
            item["_comp"] = to_float(row.get("comp_sec"))
            item["_comm"] = to_float(row.get("comm_sec"), 0.0)
            item["_total"] = to_float(row.get("total_sec"))
            item["_variant"] = (row.get("variant") or "").strip()
            item["_prog"] = (row.get("prog") or "").strip()
            item["_scenario"] = (row.get("scenario") or "").strip()
            n3_rows.append(item)
        if n3_rows:
            print(f"[LOAD] {n3_path.name}: {len(n3_rows)} rows")
            all_rows.extend(n3_rows)

    print("\n========== GENERATE FIGURES ==========")

    # Timing decomposition can be plotted for any scenario with measured phases.
    for prefix in ("kb1", "kb2", "kb3", "kb4", "kb5"):
        if family_rows[prefix]:
            plot_stacked_timing(family_rows[prefix], prefix.upper())

    n3_rows = [r for r in all_rows if r["_source"] == "N3_06"]
    if n3_rows:
        plot_stacked_timing(n3_rows, "N3_06")

    baselines = find_seq_baselines(all_rows)
    if baselines:
        print(f"[INFO] Sequential ijk baselines found for n={sorted(baselines)}")
    else:
        print("[INFO] No sequential ijk baseline available yet")
    plot_speedup_efficiency(family_rows["kb1"], baselines)

    plot_kb2(family_rows["kb2"])
    plot_kb4(family_rows["kb4"])
    plot_kb5(family_rows["kb5"])

    write_summary(all_rows)

    print("\n========== DONE ==========")
    print(f"Figures directory: {FIG_DIR}")
    print("No benchmark was executed by this script.")


if __name__ == "__main__":
    main()
