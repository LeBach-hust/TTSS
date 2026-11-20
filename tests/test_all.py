"""
tests/test_all.py
Bộ test tự động theo đúng kịch bản:
  - Suite dot    : DOT-01 -> DOT-06 (4 bản tích vô hướng)
  - Suite matmul : MM-01  -> MM-07  (4 bản nhân ma trận chặn)
  - Suite nb     : NB-01  -> NB-06  (2 bản nhân ma trận không chặn)
Cách chạy:
    python tests/test_all.py --blocking-only   # Chạy 8 bản chặn (DOT + MM)
    python tests/test_all.py                   # Chạy đủ 10 bản (DOT + MM + NB)
    python tests/test_all.py --suite dot       # Chỉ chạy DOT-01..DOT-06
    python tests/test_all.py --suite matmul    # Chỉ chạy MM-01..MM-07
    python tests/test_all.py --suite nb        # Chỉ chạy NB-01..NB-06
"""

import argparse
import csv
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT_DIR / "bin"
MPI_CMD = "mpiexec"
TIMEOUT_SEC = 90

IS_WINDOWS = platform.system() == "Windows"
EXE_EXT = ".exe" if IS_WINDOWS else ""

CSV_SCHEMA = [
    "prog", "variant", "n", "procs", "threads", "block", "chunk", "run",
    "init_sec", "comp_sec", "comm_sec", "total_sec", "checksum",
]

CHECKSUM_RE = re.compile(
    r"checksum\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def get_exe(name: str) -> Path:
    return BIN_DIR / f"{name}{EXE_EXT}"


def build_env(threads: int = 1) -> Dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    if IS_WINDOWS:
        msys_bins = [
            r"C:\msys64\ucrt64\bin",
            r"C:\msys64\mingw64\bin",
            r"C:\msys64\usr\bin",
        ]
        existing = [p for p in msys_bins if Path(p).is_dir()]
        if existing:
            env["PATH"] = os.pathsep.join(existing) + os.pathsep + env.get("PATH", "")
    return env


def build_cmd(
    prog: str,
    n: int,
    run_id: int = 1,
    csv_path: Optional[str] = None,
    p: int = 1,
    threads: int = 1,
    variant: Optional[str] = None,
    block: Optional[int] = None,
    chunk: Optional[int] = None,
    no_verify: bool = False,
) -> List[str]:
    exe = str(get_exe(prog))
    cmd: List[str] = []
    if p > 1 or "mpi" in prog or "hybrid" in prog:
        cmd.extend([MPI_CMD, "-n", str(p)])
    cmd.append(exe)
    cmd.extend(["-n", str(n), "-r", str(run_id)])
    if csv_path is not None:
        cmd.extend(["-o", csv_path])
    if prog in ("dot_omp", "matmul_omp"):
        cmd.extend(["-t", str(threads)])
    if variant is not None and prog in ("matmul_seq", "matmul_omp"):
        cmd.extend(["-v", variant])
    if block is not None and prog in ("matmul_seq", "matmul_omp", "cannon_mpi", "cannon_mpi_nb"):
        cmd.extend(["-b", str(block)])
    if chunk is not None and prog == "cannon_hybrid_nb":
        cmd.extend(["--chunk", str(chunk)])
    if no_verify and prog in ("dot_mpi", "dot_hybrid", "cannon_mpi", "cannon_hybrid", "cannon_mpi_nb"):
        cmd.append("--no-verify")
    return cmd


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    checksum: Optional[float]
    timed_out: bool = False
    missing_file: bool = False


def run_cmd(cmd: List[str], threads: int = 1) -> RunResult:
    exe_path = Path(cmd[3] if cmd[0] == MPI_CMD else cmd[0])
    if not exe_path.exists():
        return RunResult(-1, "", f"Không tìm thấy {exe_path.name}", None, missing_file=True)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            env=build_env(threads),
            cwd=str(ROOT_DIR),
        )
    except subprocess.TimeoutExpired:
        return RunResult(-1, "", f"TIMEOUT > {TIMEOUT_SEC}s (nghi ngờ deadlock)", None, timed_out=True)

    matches = CHECKSUM_RE.findall(proc.stdout)
    val = float(matches[-1]) if matches else None
    if val is not None and (math.isnan(val) or math.isinf(val)):
        val = None
    return RunResult(proc.returncode, proc.stdout, proc.stderr, val)


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-15)


class TestRunner:
    def __init__(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.ref_cache: Dict[Tuple[str, int], Optional[float]] = {}
        self.tc_results: List[Tuple[str, bool, str]] = []
        self.prog_status: Dict[str, bool] = {}

    def close(self) -> None:
        self.tmpdir.cleanup()

    def mark_prog(self, prog: str, ok: bool) -> None:
        self.prog_status[prog] = self.prog_status.get(prog, True) and ok

    def record_tc(self, tc_id: str, ok: bool, detail: str) -> None:
        self.tc_results.append((tc_id, ok, detail))
        status = "OK" if ok else "SAI"
        print(f"{tc_id:<8} | {status:<4} | {detail}")

    def get_ref(self, kind: str, n: int) -> Optional[float]:
        key = (kind, n)
        if key not in self.ref_cache:
            csv_file = str(self.tmp_path / f"ref_{kind}_{n}.csv")
            if kind == "dot":
                cmd = build_cmd("dot_seq", n=n, run_id=1, csv_path=csv_file)
            else:
                cmd = build_cmd("matmul_seq", n=n, run_id=1, csv_path=csv_file, variant="ijk")
            res = run_cmd(cmd, threads=1)
            self.ref_cache[key] = res.checksum if res.returncode == 0 else None
        return self.ref_cache[key]

    # BẢN TEST TÍCH VÔ HƯỚNG (DOT-01 -> DOT-06)

    def run_dot_suite(self) -> None:
        print("\nBẢN TEST TÍCH VÔ HƯỚNG (4 BẢN)")

        # DOT-01: Baseline tuần tự & warm-up (-r 0 không ghi CSV, -r 1 ghi 1 dòng đúng schema)
        csv_01 = self.tmp_path / "dot01.csv"
        r0 = run_cmd(build_cmd("dot_seq", n=1000, run_id=0, csv_path=str(csv_01)))
        no_csv_on_r0 = (r0.returncode == 0) and (not csv_01.exists() or csv_01.stat().st_size == 0)
        r1 = run_cmd(build_cmd("dot_seq", n=1000, run_id=1, csv_path=str(csv_01)))
        schema_ok, rows_cnt = False, 0
        if csv_01.exists():
            with open(csv_01, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                if rows:
                    schema_ok = (rows[0] == CSV_SCHEMA)
                    rows_cnt = len(rows) - 1
        ok = (r0.returncode == 0) and no_csv_on_r0 and (r1.returncode == 0) and schema_ok and (rows_cnt == 1)
        self.mark_prog("dot_seq", ok)
        self.record_tc("DOT-01", ok, f"r0_no_csv={no_csv_on_r0}, schema_ok={schema_ok}, rows={rows_cnt}")

        # DOT-02: Khớp checksum kích thước chia hết (N=100000)
        ref = self.get_ref("dot", 100000)
        errs = []
        configs = [
            ("dot_omp", 1, t) for t in (1, 2, 4, 8)
        ] + [
            ("dot_mpi", p, 1) for p in (1, 2, 4)
        ] + [
            ("dot_hybrid", 2, 2), ("dot_hybrid", 4, 2),
        ]
        for prog, p, t in configs:
            csv_f = str(self.tmp_path / f"dot02_{prog}_{p}_{t}.csv")
            res = run_cmd(build_cmd(prog, n=100000, run_id=1, csv_path=csv_f, p=p, threads=t), threads=t)
            c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-12)
            self.mark_prog(prog, c_ok)
            if not c_ok:
                errs.append(f"{prog}(p={p},t={t})")
        self.record_tc("DOT-02", len(errs) == 0, "Khớp 1e-12 mọi cấu hình" if not errs else f"Lỗi: {', '.join(errs)}")

        # DOT-03: Kích thước lẻ / nguyên tố
        errs = []
        for n in (100003, 500003, 1000003):
            ref = self.get_ref("dot", n)
            for prog, p, t in [("dot_omp", 1, 4), ("dot_mpi", 3, 1), ("dot_mpi", 4, 1), ("dot_hybrid", 4, 2)]:
                csv_f = str(self.tmp_path / f"dot03_{prog}_{n}_{p}.csv")
                res = run_cmd(build_cmd(prog, n=n, run_id=1, csv_path=csv_f, p=p, threads=t), threads=t)
                c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-12)
                self.mark_prog(prog, c_ok)
                if not c_ok:
                    errs.append(f"{prog}(n={n},p={p})")
        self.record_tc("DOT-03", len(errs) == 0, "Khớp 1e-12 với N nguyên tố" if not errs else f"Lỗi: {', '.join(errs)}")

        # DOT-04: Biên nhỏ (N=1 với p=1,4; N=3 với p=4,t=2)
        errs = []
        edge_cases = [
            ("dot_omp", 1, 1, 2), ("dot_mpi", 1, 1, 1), ("dot_mpi", 1, 4, 1),
            ("dot_hybrid", 1, 4, 2), ("dot_mpi", 3, 4, 1), ("dot_hybrid", 3, 4, 2),
        ]
        for prog, n, p, t in edge_cases:
            ref = self.get_ref("dot", n)
            csv_f = str(self.tmp_path / f"dot04_{prog}_{n}_{p}.csv")
            res = run_cmd(build_cmd(prog, n=n, run_id=1, csv_path=csv_f, p=p, threads=t), threads=t)
            c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-12)
            self.mark_prog(prog, c_ok)
            if not c_ok:
                errs.append(f"{prog}(n={n},p={p})")
        self.record_tc("DOT-04", len(errs) == 0, "Không crash ở biên N<p" if not errs else f"Lỗi: {', '.join(errs)}")

        # DOT-05: Cờ --no-verify trên dot_mpi và dot_hybrid
        errs = []
        for prog, p, t in [("dot_mpi", 4, 1), ("dot_hybrid", 4, 2)]:
            csv_f = self.tmp_path / f"dot05_{prog}.csv"
            res = run_cmd(build_cmd(prog, n=100003, run_id=1, csv_path=str(csv_f), p=p, threads=t, no_verify=True), threads=t)
            no_verify_printed = ("verify]" not in res.stdout.lower())
            csv_written = csv_f.exists() and csv_f.stat().st_size > 0
            c_ok = (res.returncode == 0 and no_verify_printed and csv_written)
            self.mark_prog(prog, c_ok)
            if not c_ok:
                errs.append(f"{prog}(rc={res.returncode}, no_ver={no_verify_printed}, csv={csv_written})")
        self.record_tc("DOT-05", len(errs) == 0, "--no-verify hoạt động & ghi CSV" if not errs else f"Lỗi: {', '.join(errs)}")

        # DOT-06: Tham số không hợp lệ (Negative)
        errs = []
        neg_cases = [
            ("dot_seq", build_cmd("dot_seq", n=0, run_id=1)),
            ("dot_seq", build_cmd("dot_seq", n=100, run_id=-1)),
            ("dot_omp", build_cmd("dot_omp", n=0, run_id=1, threads=2)),
            ("dot_omp", build_cmd("dot_omp", n=100, run_id=1, threads=0)),
            ("dot_omp", build_cmd("dot_omp", n=100, run_id=-1, threads=2)),
        ]
        for prog, cmd in neg_cases:
            res = run_cmd(cmd)
            c_ok = (res.returncode != 0)
            self.mark_prog(prog, c_ok)
            if not c_ok:
                errs.append(" ".join(cmd))
        self.record_tc("DOT-06", len(errs) == 0, "Thoát mã != 0 khi tham số sai" if not errs else f"Lỗi: {len(errs)} ca")

    #BẢN TEST NHÂN MA TRẬN BẢN CHẶN (MM-01 -> MM-07)
    
    def run_matmul_suite(self) -> None:
        print("\nBẢN TEST NHÂN MA TRẬN BẢN CHẶN (4 BẢN)")

        # MM-01: matmul_seq mọi biến thể vòng lặp & SIMD (N=128)
        ref128 = self.get_ref("matmul", 128)
        errs = []
        for v, blocks in [("ijk", [16]), ("kij", [16]), ("tile_nosimd", [16, 32, 64]), ("tile", [16, 32, 64])]:
            for b in blocks:
                csv_f = str(self.tmp_path / f"mm01_{v}_{b}.csv")
                res = run_cmd(build_cmd("matmul_seq", n=128, csv_path=csv_f, variant=v, block=b))
                c_ok = (res.returncode == 0 and res.checksum is not None and ref128 is not None and rel_err(res.checksum, ref128) < 1e-11)
                self.mark_prog("matmul_seq", c_ok)
                if not c_ok:
                    errs.append(f"{v}(b={b})")
        self.record_tc("MM-01", len(errs) == 0, "Khớp 1e-11 mọi variant seq" if not errs else f"Lỗi: {', '.join(errs)}")

        # MM-02: matmul_omp scale số luồng & variants (N=128, 256)
        errs = []
        for n in (128, 256):
            ref = self.get_ref("matmul", n)
            for t in (1, 2, 4, 8):
                for v, b in [("ijk", 16), ("ikj", 16), ("kij", 16), ("tile_nosimd", 32), ("tile", 32), ("tile_2d", 16)]:
                    csv_f = str(self.tmp_path / f"mm02_{n}_{t}_{v}_{b}.csv")
                    res = run_cmd(build_cmd("matmul_omp", n=n, csv_path=csv_f, threads=t, variant=v, block=b), threads=t)
                    c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-11)
                    self.mark_prog("matmul_omp", c_ok)
                    if not c_ok:
                        errs.append(f"n={n},t={t},v={v}")
        self.record_tc("MM-02", len(errs) == 0, "Khớp 1e-11 mọi cấu hình OMP" if not errs else f"Lỗi: {len(errs)} ca ({errs[0]})")

        # MM-03: Biên tiling (N không chia hết block, N < b)
        errs = []
        for n, b in [(33, 16), (100, 32), (100, 64), (15, 32)]:
            ref = self.get_ref("matmul", n)
            for prog, v, t in [("matmul_seq", "tile", 1), ("matmul_omp", "tile_2d", 4)]:
                csv_f = str(self.tmp_path / f"mm03_{prog}_{n}_{b}.csv")
                res = run_cmd(build_cmd(prog, n=n, csv_path=csv_f, threads=t, variant=v, block=b), threads=t)
                c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-11)
                self.mark_prog(prog, c_ok)
                if not c_ok:
                    errs.append(f"{prog}(n={n},b={b})")
        self.record_tc("MM-03", len(errs) == 0, "Biên tiling khớp chuẩn" if not errs else f"Lỗi: {', '.join(errs)}")

        # MM-04: Cannon blocking chia hết lưới q = sqrt(p)
        errs = []
        mm04_cases = [(1, 64), (4, 64), (4, 256), (9, 72), (9, 288)]
        for p, n in mm04_cases:
            ref = self.get_ref("matmul", n)
            for b in (0, 16, 32):
                csv_f = str(self.tmp_path / f"mm04_{p}_{n}_{b}.csv")
                res = run_cmd(build_cmd("cannon_mpi", n=n, csv_path=csv_f, p=p, block=b, no_verify=True))
                c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-11)
                self.mark_prog("cannon_mpi", c_ok)
                if not c_ok:
                    errs.append(f"p={p},n={n},b={b}")
        self.record_tc("MM-04", len(errs) == 0, "Cannon MPI chia hết lưới OK" if not errs else f"Lỗi: {', '.join(errs)}")

        # MM-05: Cannon blocking có Padding / Unpadding (gồm biên N=2, p=9)
        errs = []
        pad_cases = [(4, 17), (4, 65), (4, 100), (9, 17), (9, 65), (9, 100), (9, 2)]
        for p, n in pad_cases:
            ref = self.get_ref("matmul", n)
            for prog, t, tol in [("cannon_mpi", 1, 1e-11), ("cannon_hybrid", 2 if p == 4 else 1, 1e-9)]:
                csv_f = str(self.tmp_path / f"mm05_{prog}_{p}_{n}.csv")
                res = run_cmd(build_cmd(prog, n=n, csv_path=csv_f, p=p, threads=t, no_verify=True), threads=t)
                c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < tol)
                self.mark_prog(prog, c_ok)
                if not c_ok:
                    errs.append(f"{prog}(p={p},n={n})")
        self.record_tc("MM-05", len(errs) == 0, "Padding/Unpadding khớp chuẩn" if not errs else f"Lỗi: {', '.join(errs)}")

        # MM-06: Cannon Hybrid MPI+OpenMP & thread safety
        errs = []
        for n in (64, 100, 256):
            ref = self.get_ref("matmul", n)
            for p, t in [(1, 2), (4, 1), (4, 2), (4, 3), (9, 1)]:
                csv_f = str(self.tmp_path / f"mm06_{n}_{p}_{t}.csv")
                res = run_cmd(build_cmd("cannon_hybrid", n=n, csv_path=csv_f, p=p, threads=t, no_verify=True), threads=t)
                c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-9)
                self.mark_prog("cannon_hybrid", c_ok)
                if not c_ok:
                    errs.append(f"n={n},p={p},t={t}")
        self.record_tc("MM-06", len(errs) == 0, "Cannon Hybrid khớp 1e-9 mọi (p,t)" if not errs else f"Lỗi: {', '.join(errs)}")

        # MM-07: Negative cases (p không chính phương, N=0, -b 0, -t 0)
        errs = []
        neg_cases = [
            ("cannon_mpi",    build_cmd("cannon_mpi", n=64, p=2)),
            ("cannon_mpi",    build_cmd("cannon_mpi", n=64, p=3)),
            ("cannon_mpi",    build_cmd("cannon_mpi", n=64, p=5)),
            ("cannon_mpi",    build_cmd("cannon_mpi", n=0,  p=4)),
            ("cannon_hybrid", build_cmd("cannon_hybrid", n=64, p=2)),
            ("cannon_hybrid", build_cmd("cannon_hybrid", n=0,  p=4)),
            ("matmul_seq",    build_cmd("matmul_seq", n=64, variant="tile", block=0)),
            ("matmul_omp",    build_cmd("matmul_omp", n=64, threads=0)),
        ]
        for prog, cmd in neg_cases:
            res = run_cmd(cmd)
            c_ok = (res.returncode != 0 and not res.timed_out)
            self.mark_prog(prog, c_ok)
            if not c_ok:
                errs.append(f"{prog}({' '.join(cmd[-4:])})")
        self.record_tc("MM-07", len(errs) == 0, "Báo lỗi & thoát mã != 0" if not errs else f"Lỗi: {', '.join(errs)}")

    
    #BẢN TEST NHÂN MA TRẬN KHÔNG CHẶN (NB-01 -> NB-06)

    def run_nb_suite(self) -> None:
        print("\nBẢN TEST NHÂN MA TRẬN KHÔNG CHẶN (2 BẢN)")

        # NB-01: cannon_mpi_nb chia hết lưới q
        errs = []
        for p, n in [(1, 64), (4, 64), (4, 256), (9, 72), (9, 288)]:
            ref = self.get_ref("matmul", n)
            for b in (0, 16, 32):
                csv_f = str(self.tmp_path / f"nb01_{p}_{n}_{b}.csv")
                res = run_cmd(build_cmd("cannon_mpi_nb", n=n, csv_path=csv_f, p=p, block=b, no_verify=True))
                c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-11)
                self.mark_prog("cannon_mpi_nb", c_ok)
                if not c_ok:
                    errs.append(f"p={p},n={n},b={b}")
        self.record_tc("NB-01", len(errs) == 0, "cannon_mpi_nb chia hết lưới OK" if not errs else f"Lỗi: {', '.join(errs)}")

        # NB-02: cannon_mpi_nb Padding / Unpadding
        errs = []
        for p, n in [(4, 17), (4, 65), (4, 100), (9, 17), (9, 65), (9, 100), (9, 2)]:
            ref = self.get_ref("matmul", n)
            csv_f = str(self.tmp_path / f"nb02_{p}_{n}.csv")
            res = run_cmd(build_cmd("cannon_mpi_nb", n=n, csv_path=csv_f, p=p, no_verify=True))
            c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-11)
            self.mark_prog("cannon_mpi_nb", c_ok)
            if not c_ok:
                errs.append(f"p={p},n={n}")
        self.record_tc("NB-02", len(errs) == 0, "cannon_mpi_nb padding OK" if not errs else f"Lỗi: {', '.join(errs)}")

        # NB-03: Stress test lặp 5 lần liên tiếp (-r 1..5), cùng checksum, không deadlock
        errs = []
        for p, n in [(4, 512), (4, 576), (9, 576)]:
            vals = []
            for r in range(1, 6):
                csv_f = str(self.tmp_path / f"nb03_{p}_{n}.csv")
                res = run_cmd(build_cmd("cannon_mpi_nb", n=n, run_id=r, csv_path=csv_f, p=p, no_verify=True))
                if res.returncode != 0 or res.checksum is None:
                    errs.append(f"p={p},n={n},r={r}")
                    break
                vals.append(res.checksum)
            if len(vals) == 5 and any(rel_err(v, vals[0]) >= 1e-12 for v in vals[1:]):
                errs.append(f"lệch checksum giữa 5 lần (p={p},n={n})")
        ok_nb03 = (len(errs) == 0)
        self.mark_prog("cannon_mpi_nb", ok_nb03)
        self.record_tc("NB-03", ok_nb03, "5 lần liên tiếp không treo, cùng checksum" if ok_nb03 else f"Lỗi: {', '.join(errs)}")

        # NB-04: Negative case p không chính phương (p=2, 3, 6)
        errs = []
        for p in (2, 3, 6):
            res = run_cmd(build_cmd("cannon_mpi_nb", n=64, p=p))
            c_ok = (res.returncode != 0 and not res.timed_out)
            self.mark_prog("cannon_mpi_nb", c_ok)
            if not c_ok:
                errs.append(f"p={p}")
        self.record_tc("NB-04", len(errs) == 0, "Báo lỗi p không chính phương" if not errs else f"Lỗi: {', '.join(errs)}")

        # NB-05: cannon_hybrid_nb tính đúng đắn & chunk polling
        errs = []
        for n in (64, 100, 256):
            ref = self.get_ref("matmul", n)
            for chunk in (1, 8, 16, 32, 128):
                csv_f = str(self.tmp_path / f"nb05_{n}_{chunk}.csv")
                res = run_cmd(build_cmd("cannon_hybrid_nb", n=n, csv_path=csv_f, p=4, threads=2, chunk=chunk), threads=2)
                c_ok = (res.returncode == 0 and res.checksum is not None and ref is not None and rel_err(res.checksum, ref) < 1e-9)
                self.mark_prog("cannon_hybrid_nb", c_ok)
                if not c_ok:
                    errs.append(f"n={n},c={chunk}")
        self.record_tc("NB-05", len(errs) == 0, "cannon_hybrid_nb khớp 1e-9" if not errs else f"Lỗi/Chưa cài đặt ({len(errs)} ca)")

        # NB-06: cannon_hybrid_nb stress test overlap + OpenMP (lặp 5 lần)
        errs = []
        for t in (2, 3):
            for n in (512, 1024):
                vals = []
                for r in range(1, 6):
                    csv_f = str(self.tmp_path / f"nb06_{t}_{n}.csv")
                    res = run_cmd(build_cmd("cannon_hybrid_nb", n=n, run_id=r, csv_path=csv_f, p=4, threads=t, chunk=16), threads=t)
                    if res.returncode != 0 or res.checksum is None:
                        errs.append(f"t={t},n={n},r={r}")
                        break
                    vals.append(res.checksum)
                if len(vals) == 5 and any(rel_err(v, vals[0]) >= 1e-9 for v in vals[1:]):
                    errs.append(f"lệch checksum (t={t},n={n})")
        ok_nb06 = (len(errs) == 0)
        self.mark_prog("cannon_hybrid_nb", ok_nb06)
        self.record_tc("NB-06", ok_nb06, "5 lần overlap không data race" if ok_nb06 else f"Lỗi/Chưa cài đặt ({len(errs)} ca)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bộ test tự động theo kịch bản DOT / MM / NB.")
    parser.add_argument("--blocking-only", action="store_true", help="Chỉ chạy 8 bản chặn (DOT + MM, mốc N1-07).")
    parser.add_argument("--suite", choices=["all", "dot", "matmul", "nb"], default="all", help="Chọn nhóm test muốn chạy.")
    args = parser.parse_args()

    runner = TestRunner()
    try:
        print(f"{'MÃ TC':<8} | {'KQ':<4} | CHI TIẾT")
        print("-" * 78)

        if args.blocking_only:
            runner.run_dot_suite()
            runner.run_matmul_suite()
        elif args.suite == "dot":
            runner.run_dot_suite()
        elif args.suite == "matmul":
            runner.run_matmul_suite()
        elif args.suite == "nb":
            runner.run_nb_suite()
        else:
            runner.run_dot_suite()
            runner.run_matmul_suite()
            runner.run_nb_suite()

        print("\n" + "=" * 50)
        print("TỔNG KẾT THEO CHƯƠNG TRÌNH")
        print("=" * 50)
        ok_progs = 0
        for prog, passed in runner.prog_status.items():
            if passed:
                ok_progs += 1
            print(f"  {prog:<20} : {'OK' if passed else 'SAI'}")

        total_progs = len(runner.prog_status)
        ok_tcs = sum(1 for _, ok, _ in runner.tc_results if ok)
        total_tcs = len(runner.tc_results)
        print("-" * 50)
        print(f"KẾT QUẢ TC  : {ok_tcs}/{total_tcs} kịch bản OK")
        print(f"KẾT QUẢ BẢN : {ok_progs}/{total_progs} bản OK")

        return 0 if (ok_tcs == total_tcs and ok_progs == total_progs) else 1
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())