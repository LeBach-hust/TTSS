#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace common {

// Seed co dinh cho viec sinh du lieu vector/ma tran (de dam bao ket qua lap lai duoc)
constexpr uint64_t SEED_VEC_X = 1001ULL;
constexpr uint64_t SEED_VEC_Y = 2002ULL;
constexpr uint64_t SEED_MAT_A = 3003ULL;
constexpr uint64_t SEED_MAT_B = 4004ULL;

// Lay thoi gian thuc (giay) dung cho ban tuan tu / OpenMP
inline double wall_time_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Sinh du lieu vector/ma tran ngau nhien trong khoang [-1.0, 1.0] voi seed co dinh
inline void init_data(std::vector<double>& data, size_t total_elements, uint64_t seed) {
    data.resize(total_elements);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < total_elements; ++i) {

inline constexpr std::uint32_t kDefaultSeed = 20251115;

// Tinh checksum cua vector/ma tran
inline double compute_checksum(const double* data, size_t total_elements) {
    double sum = 0.0;
    for (size_t i = 0; i < total_elements; ++i) {
        double weight = 1.0 + static_cast<double>(i % 97) * 0.01;
        sum += data[i] * weight;
    }
    return sum;
}

inline double compute_checksum(const std::vector<double>& data) {
    return compute_checksum(data.data(), data.size());
}

// Tinh sai so tuong doi giua 2 gia tri (val, ref)
inline double relative_error(double val, double ref) {
    double denom = std::max(std::abs(ref), 1e-15);
    return std::abs(val - ref) / denom;
}

} // namespace common

#endif // COMMON_H