"""
scenarios/kb4.py

Kich ban 4 (N3-10): tan suat goi MPI_Testall (khong goi / moi 64 / 16 / 4
hang), n = 2304 - phu thuoc N3-05 (cannon_hybrid_nb.cpp).

TRANG THAI: NOT_IMPLEMENTED.
Da kiem tra truc tiep src/matmul/cannon_hybrid_nb.cpp - hien chi la:
    // TODO: implement chunked overlap and configurable polling frequency.
    int main() { return 0; }
Checklist danh dau N3-05 la "Xong" nhung Nghia da xac nhan day la ghi nham,
thuc te CHUA lam. Theo dung yeu cau de bai: khong duoc dua bien the chua
trien khai vao benchmark chinh thuc, nen o day khong tra ve cau hinh nao de
chay that - chi khai bao NOT_IMPLEMENTED de runner bao cao ro rang, tranh
hieu lam la "khong co du lieu vi loi chay" thanh "co du lieu nhung = 0".

Khi N3-05 that su xong (binary nhan duoc --chunk va chay chuong trinh that),
chi can thay implemented=False -> True va dien args/processes/threads tuong
ung, khong can sua engine/runner.
"""

from __future__ import annotations

from utils.command_builder import make_config, RunConfig

NOTE: str | None = (
    "KB4 bi khoa: cannon_hybrid_nb.cpp hien la stub (int main(){return 0;}), "
    "N3-05 chua thuc su xong du checklist ghi 'Xong'. Khong co cau hinh nao "
    "duoc chay cho den khi N3-05 hoan thanh that."
)

MATRIX_SIZE = 2304
TESTALL_FREQUENCIES = ("none", 64, 16, 4)  # "none" = khong goi MPI_Testall


def build() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for freq in TESTALL_FREQUENCIES:
        configs.append(make_config(
            f"Hybrid-nb-testall-{freq}-n{MATRIX_SIZE}", "cannon_hybrid_nb",
            [],  # se dien khi N3-05 xong: vd ["-n", str(MATRIX_SIZE), "--chunk", str(freq)]
            mode="hybrid", processes=4, threads=2,
            implemented=False,
            skip_reason="cannon_hybrid_nb.cpp la stub (N3-05 chua xong)",
        ))
    return configs
