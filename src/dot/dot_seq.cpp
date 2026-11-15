#include <iostream>
#include <vector>
#include "common.h"

double calc_dot_seq(const std::vector<double>& x, const std::vector<double>& y, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

int main(int argc, char* argv[]) {
    size_t n = 10000000;
    if (argc > 1) {
        n = std::stoull(argv[1]);
    }

    // Khoi tao 2 vector x, y voi seed co dinh
    double t0 = common::wall_time_sec();
    std::vector<double> x, y;
    common::init_data(x, n, common::SEED_VEC_X);
    common::init_data(y, n, common::SEED_VEC_Y);
    double init_sec = common::wall_time_sec() - t0;

    // Tinh dot product lan dau de co gia tri tham chieu (checksum)
    double res = calc_dot_seq(x, y, n);

    // Chay 5 lan tinh dot product va ghi ket qua vao CSV + in ra console
    for (int run = 1; run <= 5; run++) {
        double start = common::wall_time_sec();
        res = calc_dot_seq(x, y, n);
        double comp_sec = common::wall_time_sec() - start;

        common::RunRecord rec;
        rec.prog = "dot_seq";
        rec.variant = "seq";
        rec.n = n;
        rec.procs = 1;
        rec.threads = 1;
        rec.block = 0;
        rec.run = run;
        rec.init_sec = init_sec;
        rec.comp_sec = comp_sec;
        rec.comm_sec = 0.0;
        rec.total_sec = comp_sec;
        rec.checksum = res;

        common::append_output_csv(rec);
        common::print_run_summary(rec);
    }

    return 0;
}