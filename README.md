# matmul-locality-cannon

Dự án nghiên cứu và đánh giá hiệu năng của các thuật toán tích vô hướng và nhân ma trận trên các mô hình tuần tự, OpenMP, MPI và hybrid MPI + OpenMP.

## Cấu trúc
- `include/`: giao diện dùng chung, sinh dữ liệu và đo thời gian.
- `src/dot/`: các phiên bản tích vô hướng.
- `src/matmul/`: nhân ma trận tuần tự, OpenMP và các biến thể Cannon.
- `tests/`: kiểm tra kết quả/checksum giữa các phiên bản.
- `bench/`: chạy benchmark và vẽ đồ thị.
- `results/`: kết quả đo, thông tin máy và hình vẽ.
- `report/`, `slide/`: báo cáo và slide.

## Build
Yêu cầu C++17, GNU Make; cài OpenMP cho các bản OMP và MPI (ví dụ MS-MPI trên Windows hoặc MPICH/Open MPI trên Linux/macOS).

```bash
make seq
make omp
make mpi
make hybrid
```

Trên Windows có thể dùng `build.bat` sau khi cấu hình PATH cho MSYS2 e MS-MPI.

> Các file nguồn hiện là khung ban đầu để thiết lập cấu trúc; cần bổ sung thuật toán và môi trường MPI/OpenMP trước khi chạy build/test/benchmark.

## Kiểm thử và benchmark
```bash
python tests/test_all.py
python bench/run_all.py
python bench/plot.py
```

Các kết quả thực nghiệm thực tế cần được ghi vào `results/`; không đưa số liệu giả vào báo cáo.
