
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../include/common.h"
#include "../../include/cannon_grid.h"
#include "../../include/timing.h"

namespace {

bool parse_positive(const std::string& s, size_t& out) {
    if (s.empty() || s[0] == '-') return false;

    try {
        size_t pos = 0;
        unsigned long long value = std::stoull(s, &pos);

        if (pos != s.size() || value == 0) return false;

        out = static_cast<size_t>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}


void multiply_rows(
    const double* A,
    const double* B,
    double* C,
    int b,
    int row_begin,
    int row_end
) {
    for (int i = row_begin; i < row_end; ++i) {
        for (int k = 0; k < b; ++k) {
            const double a_ik = A[i * b + k];

            for (int j = 0; j < b; ++j) {
                C[i * b + j] += a_ik * B[k * b + j];
            }
        }
    }
}

// Nhân ma trận tuần tự để kiểm tra kết quả.
void matmul_sequential(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    int n
) {
    C.assign(static_cast<size_t>(n) * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const double a_ik = A[static_cast<size_t>(i) * n + k];

            for (int j = 0; j < n; ++j) {
                C[static_cast<size_t>(i) * n + j] +=
                    a_ik * B[static_cast<size_t>(k) * n + j];
            }
        }
    }
}

} 


int main(int argc, char** argv) {

    int provided = MPI_THREAD_SINGLE;

    MPI_Init_thread(
        &argc,
        &argv,
        MPI_THREAD_FUNNELED,
        &provided
    );

    int world_rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            std::cerr
                << "[ERROR] MPI does not support MPI_THREAD_FUNNELED\n";
        }

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t N = 288;
    size_t chunk_value = 32;

    int run_id = 1;
    std::string csv_path;

    bool verify = true;
    bool bad_arg = false;

    int positional = 0;

    for (int i = 1; i < argc; ++i) {

        std::string arg = argv[i];

        if (arg == "-n" || arg == "--n") {

            if (i + 1 >= argc ||
                !parse_positive(argv[++i], N)) {
                bad_arg = true;
            }

        } else if (arg == "-c" || arg == "--chunk") {

            if (i + 1 >= argc ||
                !parse_positive(argv[++i], chunk_value)) {
                bad_arg = true;
            }

        } else if (arg == "-r" || arg == "--run") {

            size_t value = 0;

            if (i + 1 >= argc ||
                !parse_positive(argv[++i], value) ||
                value > static_cast<size_t>(INT_MAX)) {
                bad_arg = true;
            } else {
                run_id = static_cast<int>(value);
            }

        } else if (arg == "-o" || arg == "--csv") {

            if (i + 1 >= argc) {
                bad_arg = true;
            } else {
                csv_path = argv[++i];
            }

        } else if (arg == "--no-verify") {

            verify = false;

        } else if (arg == "--verify") {

            verify = true;

        } else if (!arg.empty() && arg[0] != '-') {

            ++positional;

            size_t value = 0;

            if (positional == 1) {

                if (!parse_positive(arg, N)) {
                    bad_arg = true;
                }

            } else if (positional == 2) {

                if (!parse_positive(arg, chunk_value)) {
                    bad_arg = true;
                }

            } else if (positional == 3) {

                if (!parse_positive(arg, value) ||
                    value > static_cast<size_t>(INT_MAX)) {
                    bad_arg = true;
                } else {
                    run_id = static_cast<int>(value);
                }

            } else if (positional == 4) {

                csv_path = arg;

            } else {

                bad_arg = true;
            }

        } else {

            bad_arg = true;
        }
    }

    if (bad_arg || N > static_cast<size_t>(INT_MAX) ||
        chunk_value > static_cast<size_t>(INT_MAX)) {

        if (world_rank == 0) {
            std::cerr
                << "[ERROR] Invalid arguments.\n"
                << "Usage: mpiexec -n 4 cannon_hybrid_nb "
                << "-n <N> -c <chunk> [-r run] [-o csv]\n";
        }

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int n = static_cast<int>(N);
    const int requested_chunk = static_cast<int>(chunk_value);
    const int threads = omp_get_max_threads();

    cannon::CannonGrid grid;

    if (!cannon::init_grid(grid, n, MPI_COMM_WORLD)) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int b = grid.b;
    const int count = b * b;

    // Chunk thực tế không vượt quá số hàng của block.
    const int chunk = std::min(requested_chunk, b);

    std::vector<double> global_A;
    std::vector<double> global_B;
    std::vector<double> global_C;

    if (grid.rank == 0) {

        const size_t total_elements =
            static_cast<size_t>(n) * n;

        common::init_data(
            global_A,
            total_elements,
            common::SEED_MAT_A
        );

        common::init_data(
            global_B,
            total_elements,
            common::SEED_MAT_B
        );
    }

    const size_t block_elements =
        static_cast<size_t>(b) * b;

    std::vector<double> buf_A0(block_elements, 0.0);
    std::vector<double> buf_A1(block_elements, 0.0);

    std::vector<double> buf_B0(block_elements, 0.0);
    std::vector<double> buf_B1(block_elements, 0.0);

    std::vector<double> local_C(block_elements, 0.0);


    PhaseTimer timer;
    timer.reset();

    MPI_Barrier(grid.comm);

    timer.start_init();

    cannon::scatter_matrix(grid, global_A, buf_A0);
    cannon::scatter_matrix(grid, global_B, buf_B0);

    timer.stop_init();

    MPI_Barrier(grid.comm);

    timer.start();

    timer.start_comm();

    cannon::initial_align(grid, buf_A0, buf_B0);

    timer.stop_comm();

    double* cur_A = buf_A0.data();
    double* next_A = buf_A1.data();

    double* cur_B = buf_B0.data();
    double* next_B = buf_B1.data();

    for (int step = 0; step < grid.q; ++step) {

        const bool has_shift = (step + 1 < grid.q);

        MPI_Request requests[4] = {
            MPI_REQUEST_NULL,
            MPI_REQUEST_NULL,
            MPI_REQUEST_NULL,
            MPI_REQUEST_NULL
        };

        if (has_shift) {

            const int tag_A = 300 + step * 2;
            const int tag_B = 301 + step * 2;

            const double comm_start = MPI_Wtime();

            MPI_Irecv(
                next_A,
                count,
                MPI_DOUBLE,
                grid.right,
                tag_A,
                grid.comm,
                &requests[0]
            );

            MPI_Irecv(
                next_B,
                count,
                MPI_DOUBLE,
                grid.down,
                tag_B,
                grid.comm,
                &requests[1]
            );

            MPI_Isend(cur_A,  count, MPI_DOUBLE, grid.left,tag_A, grid.comm, &requests[2]);
            MPI_Isend(cur_B, count, MPI_DOUBLE,  grid.up, tag_B, grid.comm, &requests[3]);
            timer.comm_sec += MPI_Wtime() - comm_start;
        }

        const double comp_start = MPI_Wtime();

        #pragma omp parallel
        {
            for (int row_begin = 0;
                 row_begin < b;
                 row_begin += chunk) {

                const int row_end =
                    std::min(row_begin + chunk, b);
                #pragma omp for schedule(static)
                for (int i = row_begin; i < row_end; ++i) {

                    for (int k = 0; k < b; ++k) {

                        const double a_ik =
                            cur_A[i * b + k];

                        for (int j = 0; j < b; ++j) {

                            local_C[i * b + j] +=
                                a_ik * cur_B[k * b + j];
                        }
                    }
                }
                #pragma omp master
                {
                    if (has_shift) {

                        int completed = 0;

                        const double poll_start = MPI_Wtime();

                        MPI_Testall(
                            4,
                            requests,
                            &completed,
                            MPI_STATUSES_IGNORE
                        );

                        timer.comm_sec +=
                            MPI_Wtime() - poll_start;
                    }
                }
                #pragma omp barrier
            }
        }

        timer.comp_sec += MPI_Wtime() - comp_start;

        if (has_shift) {

            const double wait_start = MPI_Wtime();

            MPI_Waitall(
                4,
                requests,
                MPI_STATUSES_IGNORE
            );

            timer.comm_sec += MPI_Wtime() - wait_start;

            std::swap(cur_A, next_A);
            std::swap(cur_B, next_B);
        }
    }

    timer.stop();
    timer.start_init();

    cannon::gather_matrix(
        grid,
        local_C,
        global_C
    );

    timer.stop_init();

    PhaseTimer max_timer = reduce_max(timer, grid.comm);

    int exit_code = EXIT_SUCCESS;

    if (grid.rank == 0) {

        const double checksum =
            common::compute_checksum(global_C);

        common::RunRecord record;

        record.prog      = "cannon_hybrid_nb";
        record.variant   = "nb_chunked";

        record.n         = N;
        record.procs     = grid.p;
        record.threads   = threads;

        record.block     = 0;
        record.chunk     = requested_chunk;
        record.run       = run_id;

        record.init_sec  = max_timer.init_sec;
        record.comp_sec  = max_timer.comp_sec;
        record.comm_sec  = max_timer.comm_sec;
        record.total_sec = max_timer.total_sec;

        record.checksum  = checksum;

        std::cout
            << "[cannon_hybrid_nb | config]"
            << " procs=" << grid.p
            << " q=" << grid.q
            << " threads=" << threads
            << " n=" << n
            << " n_pad=" << grid.n_pad
            << " b=" << b
            << " chunk=" << requested_chunk
            << " actual_chunk=" << chunk
            << "\n";

        common::print_run_summary(record);

        std::cout
            << std::fixed
            << std::setprecision(6)
            << "[cannon_hybrid_nb | timing]"
            << " init_sec=" << max_timer.init_sec << "s"
            << " comp_sec=" << max_timer.comp_sec << "s"
            << " comm_sec=" << max_timer.comm_sec << "s"
            << " total_sec=" << max_timer.total_sec << "s"
            << "\n";

        bool pass = true;

        if (verify) {

            std::vector<double> reference_C;

            matmul_sequential(
                global_A,
                global_B,
                reference_C,
                n
            );

            const double reference_checksum =
                common::compute_checksum(reference_C);

            const double rel_err =
                common::relative_error(
                    checksum,
                    reference_checksum
                );

            double max_abs_error = 0.0;

            for (size_t i = 0; i < global_C.size(); ++i) {

                max_abs_error = std::max(
                    max_abs_error,
                    std::abs(global_C[i] - reference_C[i])
                );
            }

            pass = (rel_err < 1e-9);

            std::cout
                << std::scientific
                << std::setprecision(12)
                << "[cannon_hybrid_nb | verify]"
                << " cannon=" << checksum
                << " seq=" << reference_checksum
                << " rel_err=" << rel_err
                << " max_abs_err=" << max_abs_error
                << " status=" << (pass ? "PASS" : "FAIL")
                << "\n";
        }

        if (pass && !csv_path.empty()) {
            common::append_output_csv(
                record,
                csv_path
            );
        }
        if (!pass) {
            exit_code = EXIT_FAILURE;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, grid.comm);

    cannon::free_grid(grid);
    MPI_Finalize();
    return exit_code;
}
