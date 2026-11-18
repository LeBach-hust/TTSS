#include <iostream>
#include <vector>
#include <string>
#include "common.h"

//Thu tu ijk (khong toi uu spatial locality)
void matmul_ijk(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
    common::zero_data(C, n * n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

//Thu tu ikj (toi uu spatial locality)
void matmul_ikj(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
    common::zero_data(C, n * n);
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (size_t j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

//Thu tu kij
void matmul_kij(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
    common::zero_data(C, n * n);
    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < n; i++) {
            double a_ik = A[i * n + k];
            for (size_t j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

//Chia khoi (Tile) khong dung omp simd
void matmul_tile_nosimd(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n, size_t bs) {
    common::zero_data(C, n * n);
    for (size_t ii = 0; ii < n; ii += bs) {
        size_t i_end = std::min(ii + bs, n);
        for (size_t kk = 0; kk < n; kk += bs) {
            size_t k_end = std::min(kk + bs, n);
            for (size_t jj = 0; jj < n; jj += bs) {
                size_t j_end = std::min(jj + bs, n);
                for (size_t i = ii; i < i_end; i++) {
                    for (size_t k = kk; k < k_end; k++) {
                        double a_ik = A[i * n + k];
                        for (size_t j = jj; j < j_end; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

//Chia khoi (Tile) ket hop omp simd
void matmul_tile_simd(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n, size_t bs) {
    common::zero_data(C, n * n);
    for (size_t ii = 0; ii < n; ii += bs) {
        size_t i_end = std::min(ii + bs, n);
        for (size_t kk = 0; kk < n; kk += bs) {
            size_t k_end = std::min(kk + bs, n);
            for (size_t jj = 0; jj < n; jj += bs) {
                size_t j_end = std::min(jj + bs, n);
                for (size_t i = ii; i < i_end; i++) {
                    for (size_t k = kk; k < k_end; k++) {
                        double a_ik = A[i * n + k];
                        #pragma omp simd
                        for (size_t j = jj; j < j_end; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

void run_matmul_seq(const std::string& variant, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n, size_t bs) {
    if (variant == "ijk") {
        matmul_ijk(A, B, C, n);
    } else if (variant == "kij") {
        matmul_kij(A, B, C, n);
    } else if (variant == "tile_nosimd") {
        matmul_tile_nosimd(A, B, C, n, bs);
    } else if (variant == "tile" || variant == "tile_simd") {
        matmul_tile_simd(A, B, C, n, bs);
    } else {
        matmul_ikj(A, B, C, n);
    }
}



int main(int argc, char* argv[]) {
    size_t n = 512;
    std::string variant = "tile";
    int block_size = 64;
    int run_id = 1;
    std::string csv_path = "output.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "-n" || arg == "--n") && i + 1 < argc) {
            n = std::stoull(argv[++i]);
        } else if ((arg == "-v" || arg == "--variant") && i + 1 < argc) {
            variant = argv[++i];
        } else if ((arg == "-b" || arg == "--block") && i + 1 < argc) {
            block_size = std::stoi(argv[++i]);
        } else if ((arg == "-r" || arg == "--run") && i + 1 < argc) {
            run_id = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            std::cerr << "Usage: matmul_seq -n N -v VARIANT -b BLOCK -r RUN -o CSV\n";
            return EXIT_FAILURE;
        }
    }

    if (n == 0 || block_size <= 0 || run_id < 0 || csv_path.empty()) {
        std::cerr << "[ERROR] Invalid arguments\n";
        return EXIT_FAILURE;
    }

    size_t bs = static_cast<size_t>(block_size);
    int recorded_block =
        (variant.find("tile") != std::string::npos) ? block_size : 0;

    double t0 = common::wall_time_sec();

    std::vector<double> A, B, C;
    common::init_data(A, n * n, common::SEED_MAT_A);
    common::init_data(B, n * n, common::SEED_MAT_B);
    common::zero_data(C, n * n);

    double init_sec = common::wall_time_sec() - t0;

    double start = common::wall_time_sec();
    run_matmul_seq(variant, A, B, C, n, bs);
    double comp_sec = common::wall_time_sec() - start;

    double checksum = common::compute_checksum(C);

    common::RunRecord rec;
    rec.prog = "matmul_seq";
    rec.variant = variant;
    rec.n = n;
    rec.procs = 1;
    rec.threads = 1;
    rec.block = recorded_block;
    rec.run = run_id;
    rec.init_sec = init_sec;
    rec.comp_sec = comp_sec;
    rec.comm_sec = 0.0;
    rec.total_sec = comp_sec;
    rec.checksum = checksum;

    common::print_run_summary(rec);

    if (run_id > 0) {
        if (!common::append_output_csv(rec, csv_path)) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
