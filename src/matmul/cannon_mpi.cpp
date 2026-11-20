#include <mpi.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "../../include/common.h"
#include "../../include/cannon_grid.h"
#include "../../include/timing.h"
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

    std::vector<double> local_A(static_cast<size_t>(grid.b) * grid.b, 0.0);
    std::vector<double> local_B(static_cast<size_t>(grid.b) * grid.b, 0.0);
    std::vector<double> local_C(static_cast<size_t>(grid.b) * grid.b, 0.0);

PhaseTimer timer;
    timer.reset();

    MPI_Barrier(grid.comm);
    timer.start_init();
    cannon::scatter_matrix(grid, global_A, local_A);
    cannon::scatter_matrix(grid, global_B, local_B);
    timer.stop_init();

    MPI_Barrier(grid.comm);
    timer.start();

    timer.start_comm();
    cannon::initial_align(grid, local_A, local_B);
    timer.stop_comm();

    for (int step = 0; step < grid.q; ++step) {
        timer.start_comp();
        cannon::block_mul_add(local_A, local_B, local_C, grid.b, block_size);
        timer.stop_comp();

        if (step + 1 < grid.q) {
            timer.start_comm();
            cannon::shift_blocks_once(grid, local_A, local_B);
            timer.stop_comm();
        }
    }

    timer.stop();

    timer.start_init();
    cannon::gather_matrix(grid, local_C, global_C);
    timer.stop_init();

    PhaseTimer max_timer = reduce_max(timer, grid.comm);
    if (grid.rank == 0) {
        double checksum_mpi = common::compute_checksum(global_C);
        common::RunRecord rec;
        rec.prog = "cannon_mpi";
        rec.variant = (block_size > 0) ? "blk_tile" : "blk";
        rec.n = static_cast<size_t>(n);
        rec.procs = grid.p;
        rec.threads = 1;
        rec.block = block_size;
        rec.chunk = 0;
        rec.run = run_id;
        rec.init_sec = max_timer.init_sec;
        rec.comp_sec = max_timer.comp_sec;
        rec.comm_sec = max_timer.comm_sec;
        rec.total_sec = max_timer.total_sec;
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
                      << "[cannon_mpi | verify] n=" << n
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