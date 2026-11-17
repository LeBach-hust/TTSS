#include <mpi.h>
#include <omp.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "common.h"

namespace {

double dot_sequential(const std::vector<double>& x, const std::vector<double>& y) {
    double sum = 0.0;
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

// Tich vo huong cuc bo: cac luong OpenMP chia vong lap, gom bang reduction.
double dot_local_omp(const double* local_x, const double* local_y, int local_n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < local_n; ++i) {
        sum += local_x[i] * local_y[i];
    }
    return sum;
}

// Chi so trong mang thoi gian (sau do MPI_Reduce MAX qua moi tien trinh).
enum TimeIdx { T_INIT, T_SCATTER, T_REDUCE, T_COMP, T_COMM, T_TOTAL, T_COUNT };

} // namespace

int main(int argc, char** argv) {
    // FUNNELED: chi luong chinh goi MPI. Vung OpenMP chi tinh toan, khong goi MPI.
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, p = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "[ERROR] MPI khong ho tro MPI_THREAD_FUNNELED (provided="
                      << provided << ")\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t N = 1000003;
    int run_id = 1;
    std::string csv_path;
    bool verify = true;

    int pos_arg = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--n") && i + 1 < argc) {
            N = static_cast<size_t>(std::stoull(argv[++i]));
        } else if ((arg == "-r" || arg == "--run") && i + 1 < argc) {
            run_id = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--no-verify") {
            verify = false;
        } else if (!arg.empty() && arg[0] != '-') {
            ++pos_arg;
            if (pos_arg == 1) N = static_cast<size_t>(std::stoull(arg));
            else if (pos_arg == 2) run_id = std::stoi(arg);
            else if (pos_arg == 3) csv_path = arg;
        }
    }

    // So luong OpenMP moi tien trinh, lay tu OMP_NUM_THREADS.
    const int threads = omp_get_max_threads();

    // 1. Chia du lieu giong dot_mpi.cpp
    std::vector<int> sendcounts(p, 0);
    std::vector<int> displs(p, 0);
    size_t base_count = N / static_cast<size_t>(p);
    size_t remainder  = N % static_cast<size_t>(p);

    int offset = 0;
    for (int i = 0; i < p; ++i) {
        sendcounts[i] = static_cast<int>(base_count + (static_cast<size_t>(i) < remainder ? 1 : 0));
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_n = sendcounts[rank];
    std::vector<double> local_x(local_n, 0.0);
    std::vector<double> local_y(local_n, 0.0);

    // 2. Sinh du lieu tren rank 0 voi seed co dinh 
    std::vector<double> global_x;
    std::vector<double> global_y;
    double t_init_start = MPI_Wtime();
    if (rank == 0) {
        common::init_data(global_x, N, common::SEED_VEC_X);
        common::init_data(global_y, N, common::SEED_VEC_Y);
    }

    double t[T_COUNT] = {0.0};
    t[T_INIT] = MPI_Wtime() - t_init_start;

    // 3. Do thoi gian: Scatter, tinh toan, Reduce, tong
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    MPI_Scatterv(rank == 0 ? global_x.data() : nullptr, sendcounts.data(), displs.data(),
                 MPI_DOUBLE, local_x.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? global_y.data() : nullptr, sendcounts.data(), displs.data(),
                 MPI_DOUBLE, local_y.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    t[T_SCATTER] = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    double local_dot = dot_local_omp(local_x.data(), local_y.data(), local_n);
    t[T_COMP] = MPI_Wtime() - t0;

    double global_dot = 0.0;
    t0 = MPI_Wtime();
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t[T_REDUCE] = MPI_Wtime() - t0;

    t[T_COMM]  = t[T_SCATTER] + t[T_REDUCE];
    t[T_TOTAL] = MPI_Wtime() - t_total_start;

    double max_t[T_COUNT] = {0.0};
    MPI_Reduce(t, max_t, T_COUNT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 4. Rank 0: in, kiem tra, roi moi ghi CSV
    if (rank == 0) {
        std::cout << "[dot_hybrid | config] procs=" << p
                  << " threads=" << threads
                  << " n=" << N << "\n";

        common::RunRecord rec;
        rec.prog = "dot_hybrid";
        rec.variant = "scatterv_reduce_omp";
        rec.n = N;
        rec.procs = p;
        rec.threads = threads;
        rec.block = 0;
        rec.chunk = 0;
        rec.run= run_id;
        rec.init_sec = max_t[T_SCATTER]; 
        rec.comp_sec = max_t[T_COMP];
        rec.comm_sec = max_t[T_COMM];
        rec.total_sec = max_t[T_TOTAL];
        rec.checksum = global_dot;

        common::print_run_summary(rec);
        std::cout << std::fixed << std::setprecision(6)
                  << "[dot_hybrid | timing] scatter_sec=" << max_t[T_SCATTER] << "s"
                  << " reduce_sec=" << max_t[T_REDUCE] << "s"
                  << " init_data_sec=" << max_t[T_INIT] << "s\n";

        bool pass = true;
        if (verify) {
            double ref_dot = dot_sequential(global_x, global_y);
            double rel_err = common::relative_error(global_dot, ref_dot);
            pass = (rel_err < 1e-12);

            std::cout << std::scientific << std::setprecision(12)
                      << "[dot_hybrid | verify] hybrid=" << global_dot
                      << " seq=" << ref_dot
                      << " rel_err=" << rel_err
                      << " status=" << (pass ? "PASS" : "FAIL") << "\n";
        }

        // Chi ghi CSV khi ket qua da duoc kiem tra dat (hoac khi tat kiem tra).
        if (pass && !csv_path.empty()) {
            common::append_output_csv(rec, csv_path);
        }

        if (!pass) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}   