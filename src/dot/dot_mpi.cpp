#include <mpi.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "../../include/common.h"

namespace {

double dot_sequential(const std::vector<double>& x, const std::vector<double>& y) {
    double sum = 0.0;
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

double dot_local(const double* local_x, const double* local_y, int local_n) {
    double sum = 0.0;
    for (int i = 0; i < local_n; ++i) {
        sum += local_x[i] * local_y[i];
    }
    return sum;
}

} 

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, p = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    size_t N = 1000003; 
    int run_id = 1;
    std::string csv_path = "";
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
        } else if (arg == "--verify") {
            verify = true;
        } else if (!arg.empty() && arg[0] != '-') {
            ++pos_arg;
            if (pos_arg == 1) N = static_cast<size_t>(std::stoull(arg));
            else if (pos_arg == 2) run_id = std::stoi(arg);
            else if (pos_arg == 3) csv_path = arg;
        }
    }

    // 1. Tinh sendcounts va displs cho MPI_Scatterv (ho tro moi N le va p = 1, 2, 3, 4...)
    std::vector<int> sendcounts(p, 0);
    std::vector<int> displs(p, 0);

    size_t base_count = N / static_cast<size_t>(p);
    size_t remainder  = N % static_cast<size_t>(p);

    int offset = 0;
    for (int i = 0; i < p; ++i) {
        sendcounts[i] = static_cast<int>(base_count + (static_cast<size_t>(i) < remainder ? 1 : 0));
        displs[i]     = offset;
        offset       += sendcounts[i];
    }

    int local_n = sendcounts[rank];
    std::vector<double> local_x(local_n, 0.0);
    std::vector<double> local_y(local_n, 0.0);

    // 2. Khoi tao du lieu tren rank 0 voi seed co dinh trong common.h
    std::vector<double> global_x;
    std::vector<double> global_y;
    double t_init_start = MPI_Wtime();
    if (rank == 0) {
        common::init_data(global_x, N, common::SEED_VEC_X);
        common::init_data(global_y, N, common::SEED_VEC_Y);
    }
    double local_init_sec = MPI_Wtime() - t_init_start;

    // 3. Do rieng thoi gian Scatter (MPI_Scatterv cho x va y)
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start   = MPI_Wtime();
    double t_scatter_start = MPI_Wtime();

    MPI_Scatterv(rank == 0 ? global_x.data() : nullptr,
                 sendcounts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 local_x.data(),
                 local_n,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? global_y.data() : nullptr,
                 sendcounts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 local_y.data(),
                 local_n,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    double local_scatter_sec = MPI_Wtime() - t_scatter_start;

    // 4. Tinh tich vo huong cuc bo
    double t_comp_start = MPI_Wtime();
    double local_dot    = dot_local(local_x.data(), local_y.data(), local_n);
    double local_comp_sec = MPI_Wtime() - t_comp_start;

    // 5. Gom ket qua ve rank 0 bang MPI_Reduce
    double global_dot = 0.0;
    double t_reduce_start = MPI_Wtime();
    MPI_Reduce(&local_dot,
               &global_dot,
               1,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);
    double local_reduce_sec = MPI_Wtime() - t_reduce_start;
    double local_total_sec  = MPI_Wtime() - t_total_start;

    double local_comm_sec = local_scatter_sec + local_reduce_sec;

    double max_init_sec    = 0.0;
    double max_scatter_sec = 0.0;
    double max_reduce_sec  = 0.0;
    double max_comm_sec    = 0.0;
    double max_comp_sec    = 0.0;
    double max_total_sec   = 0.0;

    MPI_Reduce(&local_init_sec,    &max_init_sec,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_scatter_sec, &max_scatter_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_reduce_sec,  &max_reduce_sec,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_sec,    &max_comm_sec,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comp_sec,    &max_comp_sec,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_sec,   &max_total_sec,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 6. In ket qua tren rank 0 (in rieng thoi gian Scatter + print_run_summary + doi chieu tuan tu)
    if (rank == 0) {
        common::RunRecord rec;
        rec.prog      = "dot_mpi";
        rec.variant   = "scatterv_reduce";
        rec.n         = N;
        rec.procs     = p;
        rec.threads   = 1;
        rec.block     = 0;
        rec.chunk     = 0;
        rec.run       = run_id;
        rec.init_sec  = max_scatter_sec;
        rec.comp_sec  = max_comp_sec;
        rec.comm_sec  = max_comm_sec;
        rec.total_sec = max_total_sec;
        rec.checksum  = global_dot;

        common::print_run_summary(rec);
        std::cout << std::fixed << std::setprecision(6)
                  << "[dot_mpi | timing] scatter_sec=" << max_scatter_sec << "s"
                  << " reduce_sec=" << max_reduce_sec << "s"
                  << " init_data_sec=" << max_init_sec << "s\n";

        if (!csv_path.empty()) {
            common::append_output_csv(rec, csv_path);
        }

        if (verify) {
            double ref_dot = dot_sequential(global_x, global_y);
            double rel_err = common::relative_error(global_dot, ref_dot);
            bool pass      = (rel_err < 1e-12);

            std::cout << std::scientific << std::setprecision(12)
                      << "[dot_mpi | verify] mpi=" << global_dot
                      << " seq=" << ref_dot
                      << " rel_err=" << rel_err
                      << " status=" << (pass ? "PASS" : "FAIL") << "\n";

            if (!pass) {
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
