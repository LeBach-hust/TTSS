#pragma once

#include <chrono>

typedef std::chrono::steady_clock TTSSClock;

// PhaseTimer interface for initialization, computation, communication and total time.
struct PhaseTimer {
    double init_sec = 0.0;
    double comp_sec = 0.0;
    double comm_sec = 0.0;
    double total_sec = 0.0;
};
