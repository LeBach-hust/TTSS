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

void run_matmul_seq(const std::string& variant, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
    if (variant == "ijk") {
        matmul_ijk(A, B, C, n);
    } else if (variant == "kij") {
        matmul_kij(A, B, C, n);
    } else {
        matmul_ikj(A, B, C, n);
    }
}

int main(int argc, char* argv[]) {
    // Mac dinh kich thuoc ma tran n=512, thu tu tinh tich ma tran = "ikj", chay 5 lan
    size_t n = 512;
    std::string variant = "ikj";
    int num_runs = 5;

    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) variant = argv[2];
    if (argc > 3) num_runs = std::stoi(argv[3]);

    // Khoi tao 2 ma tran A, B voi seed co dinh va ma tran C = 0
    double t0 = common::wall_time_sec();
    std::vector<double> A, B, C;
    common::init_data(A, n * n, common::SEED_MAT_A);
    common::init_data(B, n * n, common::SEED_MAT_B);
    common::zero_data(C, n * n);
    double init_sec = common::wall_time_sec() - t0;

    // Chay lan dau de co gia tri tham chieu (checksum)
    run_matmul_seq(variant, A, B, C, n);

    // Chay do 5 lan tinh tich ma tran va ghi ket qua vao CSV + in ra console
    for (int run = 1; run <= num_runs; run++) {
        double start = common::wall_time_sec();
        run_matmul_seq(variant, A, B, C, n);
        double comp_sec = common::wall_time_sec() - start;

        double cs = common::compute_checksum(C);

        common::RunRecord rec;
        rec.prog = "matmul_seq";
        rec.variant = variant;
        rec.n = n;
        rec.procs = 1;
        rec.threads = 1;
        rec.block = 0;
        rec.run = run;
        rec.init_sec = init_sec;
        rec.comp_sec = comp_sec;
        rec.comm_sec = 0.0;
        rec.total_sec = comp_sec;
        rec.checksum = cs;

        common::append_output_csv(rec);
        common::print_run_summary(rec);
    }

    return 0;
}