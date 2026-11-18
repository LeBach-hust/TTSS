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

namespace {

// Doc so nguyen duong. Tra ve false neu chuoi rong, co dau '-', khong phai so, hoac bang 0.
bool parse_positive(const std::string& s, size_t& out) {
    if (s.empty() || s[0] == '-') return false;
    try {
        size_t pos = 0;
        unsigned long long v = std::stoull(s, &pos);
        if (pos != s.size() || v == 0) return false;
        out = static_cast<size_t>(v);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// Nhan khoi: C += A * B, A, B, C la b x b theo hang (row-major).
// Vong i chia cho cac luong OpenMP. Moi luong ghi cac hang C rieng, nen khong co race condition.
void block_mul_add_omp(const double* A, const double* B, double* C, int b) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < b; ++i) {
        for (int k = 0; k < b; ++k) {
            const double a_ik = A[i * b + k];
            for (int j = 0; j < b; ++j) {
                C[i * b + j] += a_ik * B[k * b + j];
            }
        }
    }
}

// Nhan tuan tu n x n lam tham chieu kiem tra. Chi rank 0 goi.
void matmul_seq_ref(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, int n) {
    const size_t sn = static_cast<size_t>(n);
    C.assign(sn * sn, 0.0);
    for (size_t i = 0; i < sn; ++i) {
        for (size_t k = 0; k < sn; ++k) {
            const double a_ik = A[i * sn + k];
            for (size_t j = 0; j < sn; ++j) {
                C[i * sn + j] += a_ik * B[k * sn + j];
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    // FUNNELED: chi luong chinh goi MPI. Vung OpenMP chi tinh toan.
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

    // 0. Doc tham so. Moi tham so sai se bao loi, khong bo qua am tham.
    size_t N = 1000;
    int run_id = 1;
    std::string csv_path;
    bool verify = true;
    bool bad_arg = false;

    int pos_arg = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--n") {
            if (i + 1 >= argc || !parse_positive(argv[++i], N)) bad_arg = true;
        } else if (arg == "-r" || arg == "--run") {
            size_t r = 0;
            if (i + 1 < argc && parse_positive(argv[++i], r)) run_id = static_cast<int>(r);
            else bad_arg = true;
        } else if (arg == "-o" || arg == "--csv") {
            if (i + 1 < argc) csv_path = argv[++i];
            else bad_arg = true;
        } else if (arg == "--no-verify") {
            verify = false;
        } else if (!arg.empty() && arg[0] != '-') {
            ++pos_arg;
            if (pos_arg == 1) {
                if (!parse_positive(arg, N)) bad_arg = true;
            } else if (pos_arg == 2) {
                size_t r = 0;
                if (parse_positive(arg, r)) run_id = static_cast<int>(r);
                else bad_arg = true;
            } else if (pos_arg == 3) {
                csv_path = arg;
            } else {
                bad_arg = true;
            }
        } else {
            bad_arg = true;  // Tuy chon khong duoc ho tro
        }
    }

    if (bad_arg || N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::cerr << "[ERROR] Tham so khong hop le. Dung: -n <N> voi 0 < N <= "
                      << INT_MAX << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int n = static_cast<int>(N);
    const int threads = omp_get_max_threads();  // Lay tu OMP_NUM_THREADS

    // 1. Tao luoi Cannon q x q (p phai la so chinh phuong)
    cannon::CannonGrid g;
    if (!cannon::init_grid(g, n, MPI_COMM_WORLD)) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // 2. Sinh A, B tren rank 0 voi seed co dinh. Viec nay khong nam trong thoi gian do.
    std::vector<double> A, B;
    if (rank == 0) {
        common::init_data(A, N * N, common::SEED_MAT_A);
        common::init_data(B, N * N, common::SEED_MAT_B);
    }

    // 3. Do thoi gian: scatter (init), dich chuyen (comm), nhan khoi (comp), gather (comm)
    std::vector<double> local_A, local_B;
    std::vector<double> local_C(static_cast<size_t>(g.b) * g.b, 0.0);
    double t_init = 0.0, t_comm = 0.0, t_comp = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_total_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    cannon::scatter_matrix(g, A, local_A);
    cannon::scatter_matrix(g, B, local_B);
    t_init = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    cannon::initial_align(g, local_A, local_B);
    t_comm += MPI_Wtime() - t0;

    for (int step = 0; step < g.q; ++step) {
        t0 = MPI_Wtime();
        block_mul_add_omp(local_A.data(), local_B.data(), local_C.data(), g.b);
        t_comp += MPI_Wtime() - t0;

        // Buoc dich sau phep nhan cuoi la thua, bo qua.
        if (step + 1 < g.q) {
            t0 = MPI_Wtime();
            cannon::shift_blocks_once(g, local_A, local_B);
            t_comm += MPI_Wtime() - t0;
        }
    }

    std::vector<double> C;
    t0 = MPI_Wtime();
    cannon::gather_matrix(g, local_C, C);
    t_comm += MPI_Wtime() - t0;

    const double t_total = MPI_Wtime() - t_total_start;

    // 4. Lay gia tri lon nhat moi pha qua moi tien trinh
    const double local_t[4] = {t_init, t_comp, t_comm, t_total};
    double max_t[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Reduce(local_t, max_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 5. Rank 0: in, kiem tra, roi moi ghi CSV
    int exit_code = EXIT_SUCCESS;
    if (rank == 0) {
        std::cout << "[cannon_hybrid | config] procs=" << p
                  << " q=" << g.q
                  << " threads=" << threads
                  << " n=" << n
                  << " n_pad=" << g.n_pad
                  << " b=" << g.b << "\n";

        const double checksum = common::compute_checksum(C);

        common::RunRecord rec;
        rec.prog      = "cannon_hybrid";
        rec.variant   = "blocking";
        rec.n         = N;
        rec.procs     = p;
        rec.threads   = threads;
        rec.block     = 0;
        rec.chunk     = 0;
        rec.run       = run_id;
        rec.init_sec  = max_t[0];
        rec.comp_sec  = max_t[1];
        rec.comm_sec  = max_t[2];
        rec.total_sec = max_t[3];
        rec.checksum  = checksum;

        common::print_run_summary(rec);
        std::cout << std::fixed << std::setprecision(6)
                  << "[cannon_hybrid | timing] init_sec=" << max_t[0] << "s"
                  << " comp_sec=" << max_t[1] << "s"
                  << " comm_sec=" << max_t[2] << "s"
                  << " total_sec=" << max_t[3] << "s\n";

        bool pass = true;
        if (verify) {
            std::vector<double> C_ref;
            matmul_seq_ref(A, B, C_ref, n);

            const double ref_checksum = common::compute_checksum(C_ref);
            const double rel_err = common::relative_error(checksum, ref_checksum);

            double max_abs = 0.0;
            for (size_t idx = 0; idx < C.size(); ++idx) {
                max_abs = std::max(max_abs, std::abs(C[idx] - C_ref[idx]));
            }

            pass = (rel_err < 1e-9);
            std::cout << std::scientific << std::setprecision(12)
                      << "[cannon_hybrid | verify] cannon=" << checksum
                      << " seq=" << ref_checksum
                      << " rel_err=" << rel_err
                      << " max_abs_err=" << max_abs
                      << " status=" << (pass ? "PASS" : "FAIL") << "\n";
        }

        // Chi ghi CSV khi ket qua da duoc kiem tra dat (hoac khi tat kiem tra).
        if (pass && !csv_path.empty()) {
            common::append_output_csv(rec, csv_path);
        }
        if (!pass) {
            exit_code = EXIT_FAILURE;
        }
    }

    cannon::free_grid(g);
    MPI_Finalize();
    return exit_code;
}