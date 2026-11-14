#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ttss {

inline constexpr std::uint32_t kDefaultSeed = 20251115;

// Deterministic input generation and CSV/checksum helpers will be implemented here.
inline std::uint64_t checksum(const std::vector<double>& values) {
    // TODO: define a stable checksum contract shared by all implementations.
    std::uint64_t result = 0;
    for (double value : values) {
        result ^= static_cast<std::uint64_t>(value * 1'000'000.0);
    }
    return result;
}

}  // namespace ttss
