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
    int run_id = 1;
    std::string csv_path = "output.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "-n" || arg == "--n") && i + 1 < argc) {
            n = std::stoull(argv[++i]);
        } else if ((arg == "-r" || arg == "--run") && i + 1 < argc) {
            run_id = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            std::cerr << "Usage: dot_seq -n N -r RUN -o CSV\n";
            return EXIT_FAILURE;
        }
    }

    if (n == 0 || run_id < 0 || csv_path.empty()) {
        std::cerr << "[ERROR] Invalid arguments\n";
        return EXIT_FAILURE;
    }

    // Khoi tao du lieu
    double t0 = common::wall_time_sec();

    std::vector<double> x, y;
    common::init_data(x, n, common::SEED_VEC_X);
    common::init_data(y, n, common::SEED_VEC_Y);

    double init_sec = common::wall_time_sec() - t0;

    // Moi lan goi chuong trinh chi tinh mot lan
    double start = common::wall_time_sec();
    double res = calc_dot_seq(x, y, n);
    double comp_sec = common::wall_time_sec() - start;

    common::RunRecord rec;
    rec.prog = "dot_seq";
    rec.variant = "seq";
    rec.n = n;
    rec.procs = 1;
    rec.threads = 1;
    rec.block = 0;
    rec.run = run_id;
    rec.init_sec = init_sec;
    rec.comp_sec = comp_sec;
    rec.comm_sec = 0.0;
    rec.total_sec = comp_sec;
    rec.checksum = res;

    common::print_run_summary(rec);

    // run = 0 la warm-up, khong ghi CSV
    if (run_id > 0) {
        if (!common::append_output_csv(rec, csv_path)) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
