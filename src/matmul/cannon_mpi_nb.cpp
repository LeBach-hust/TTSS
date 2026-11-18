#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "../../include/common.h"
#include "../../include/cannon_grid.h"

namespace {

void matmul_sequential(const std::vector<double>& A,
                       const std::vector<double>& B,
                       std::vector<double>& C,
                       int n) {
    C.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_ik = A[static_cast<size_t>(i) * n + k];
            for (int j = 0; j < n; ++j) {
                C[static_cast<size_t>(i) * n + j] += a_ik * B[static_cast<size_t>(k) * n + j];
            }
        }
    }
}

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 288;
    int block_size = 0;
    int run_id = 1;
    std::string csv_path = "";
    bool verify = true;

    int pos_arg = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--n") && i + 1 < argc) {
            n = std::stoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--block") && i + 1 < argc) {
            block_size = std::stoi(argv[++i]);
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
            if (pos_arg == 1) n = std::stoi(arg);
            else if (pos_arg == 2) block_size = std::stoi(arg);
            else if (pos_arg == 3) run_id = std::stoi(arg);
            else if (pos_arg == 4) csv_path = arg;
        }
    }

    if (n <= 0) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    cannon::CannonGrid grid;
    if (!cannon::init_grid(grid, n, MPI_COMM_WORLD)) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    std::vector<double> global_A;
    std::vector<double> global_B;
    std::vector<double> global_C;

    if (grid.rank == 0) {
        size_t total_elems = static_cast<size_t>(n) * n;
        common::init_data(global_A, total_elems, common::SEED_MAT_A);
        common::init_data(global_B, total_elems, common::SEED_MAT_B);
    }

    size_t block_elems = static_cast<size_t>(grid.b) * grid.b;
    int count = grid.b * grid.b;

    std::vector<double> buf_A0(block_elems, 0.0);
    std::vector<double> buf_A1(block_elems, 0.0);
    std::vector<double> buf_B0(block_elems, 0.0);
    std::vector<double> buf_B1(block_elems, 0.0);
    std::vector<double> local_C(block_elems, 0.0);

    MPI_Barrier(grid.comm);
    double t_total_start = MPI_Wtime();

    double t_scatter_start = MPI_Wtime();
    cannon::scatter_matrix(grid, global_A, buf_A0);
    cannon::scatter_matrix(grid, global_B, buf_B0);
    double local_init_sec = MPI_Wtime() - t_scatter_start;

    double local_comp_sec = 0.0;
    double local_comm_sec = 0.0;

    double t_align_start = MPI_Wtime();
    cannon::initial_align(grid, buf_A0, buf_B0);
    local_comm_sec += (MPI_Wtime() - t_align_start);

    double* cur_A = buf_A0.data();
    double* next_A = buf_A1.data();
    double* cur_B = buf_B0.data();
    double* next_B = buf_B1.data();

    for (int step = 0; step < grid.q; ++step) {
        MPI_Request reqs[4];
        bool has_shift = (step + 1 < grid.q);

        if (has_shift) {
            double t_post = MPI_Wtime();
            int tag_a = 300 + step * 2;
            int tag_b = 301 + step * 2;

            MPI_Irecv(next_A, count, MPI_DOUBLE, grid.right, tag_a, grid.comm, &reqs[0]);
            MPI_Irecv(next_B, count, MPI_DOUBLE, grid.down,  tag_b, grid.comm, &reqs[1]);
            MPI_Isend(cur_A,  count, MPI_DOUBLE, grid.left,  tag_a, grid.comm, &reqs[2]);
            MPI_Isend(cur_B,  count, MPI_DOUBLE, grid.up,    tag_b, grid.comm, &reqs[3]);
            local_comm_sec += (MPI_Wtime() - t_post);
        }

        double t_comp = MPI_Wtime();
        cannon::block_mul_add(cur_A, cur_B, local_C.data(), grid.b, block_size);
        local_comp_sec += (MPI_Wtime() - t_comp);

        if (has_shift) {
            double t_wait = MPI_Wtime();
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
            local_comm_sec += (MPI_Wtime() - t_wait);

            std::swap(cur_A, next_A);
            std::swap(cur_B, next_B);
        }
    }

    double t_gather_start = MPI_Wtime();
    cannon::gather_matrix(grid, local_C, global_C);
    local_init_sec += (MPI_Wtime() - t_gather_start);

    double local_total_sec = MPI_Wtime() - t_total_start;

    double max_init_sec = 0.0;
    double max_comp_sec = 0.0;
    double max_comm_sec = 0.0;
    double max_total_sec = 0.0;

    MPI_Reduce(&local_init_sec, &max_init_sec, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);
    MPI_Reduce(&local_comp_sec, &max_comp_sec, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);
    MPI_Reduce(&local_comm_sec, &max_comm_sec, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);
    MPI_Reduce(&local_total_sec, &max_total_sec, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);

    if (grid.rank == 0) {
        double checksum_mpi = common::compute_checksum(global_C);

        common::RunRecord rec;
        rec.prog = "cannon_mpi_nb";
        rec.variant = "nb";
        rec.n = static_cast<size_t>(n);
        rec.procs = grid.p;
        rec.threads = 1;
        rec.block = block_size;
        rec.chunk = 0;
        rec.run = run_id;
        rec.init_sec = max_init_sec;
        rec.comp_sec = max_comp_sec;
        rec.comm_sec = max_comm_sec;
        rec.total_sec = max_total_sec;
        rec.checksum = checksum_mpi;

        common::print_run_summary(rec);

        if (!csv_path.empty()) {
            common::append_output_csv(rec, csv_path);
        }

        if (verify) {
            std::vector<double> ref_C;
            matmul_sequential(global_A, global_B, ref_C, n);
            double checksum_seq = common::compute_checksum(ref_C);
            double rel_err = common::relative_error(checksum_mpi, checksum_seq);
            bool pass = (rel_err < 1e-11);

            std::cout << std::scientific << std::setprecision(12)
                      << "[cannon_mpi_nb | verify] n=" << n
                      << " n_pad=" << grid.n_pad
                      << " p=" << grid.p
                      << " mpi_checksum=" << checksum_mpi
                      << " seq_checksum=" << checksum_seq
                      << " rel_err=" << rel_err
                      << " status=" << (pass ? "PASS" : "FAIL") << "\n";

            if (!pass) {
                MPI_Abort(grid.comm, EXIT_FAILURE);
            }
        }
    }

    cannon::free_grid(grid);
    MPI_Finalize();
    return EXIT_SUCCESS;
}