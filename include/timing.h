#pragma once

#include <chrono>
#include <mpi.h>

// Clock dung chung cho project.
typedef std::chrono::steady_clock TTSSClock;

// Bo do thoi gian theo pha, don vi giay.
// Cac truong init_sec, comp_sec, comm_sec, total_sec ghi thang vao output.csv.
struct PhaseTimer {
    // Ket qua do thoi gian (don vi: giay).
    double init_sec = 0.0;
    double comp_sec = 0.0;
    double comm_sec = 0.0;
    double total_sec = 0.0;

private:
    // Cac moc thoi gian bat dau.
    TTSSClock::time_point total_start_;
    TTSSClock::time_point init_start_;
    TTSSClock::time_point comp_start_;
    TTSSClock::time_point comm_start_;

    // Trang thai cua tung bo dem.
    bool total_running_ = false;
    bool init_running_ = false;
    bool comp_running_ = false;
    bool comm_running_ = false;

    // Chuyen khoang thoi gian thanh giay.
    static double elapsed_seconds(
        const TTSSClock::time_point& start,
        const TTSSClock::time_point& end
    ) {
        return std::chrono::duration<double>(end - start).count();
    }

public:
    // Dua timer ve trang thai ban dau.
    void reset() {
        init_sec = 0.0;
        comp_sec = 0.0;
        comm_sec = 0.0;
        total_sec = 0.0;

        total_running_ = false;
        init_running_ = false;
        comp_running_ = false;
        comm_running_ = false;
    }

    // Bat dau do tong thoi gian.
    void start() {
        if (total_running_) return;
        total_start_ = TTSSClock::now();
        total_running_ = true;
    }

    // Ket thuc do tong thoi gian.
    void stop() {
        if (!total_running_) return;
        total_sec += elapsed_seconds(total_start_, TTSSClock::now());
        total_running_ = false;
    }

    // Bat dau do thoi gian khoi tao.
    void start_init() {
        if (init_running_) return;
        init_start_ = TTSSClock::now();
        init_running_ = true;
    }

    // Ket thuc do thoi gian khoi tao.
    void stop_init() {
        if (!init_running_) return;
        init_sec += elapsed_seconds(init_start_, TTSSClock::now());
        init_running_ = false;
    }

    // Bat dau do thoi gian tinh toan.
    void start_comp() {
        if (comp_running_) return;
        comp_start_ = TTSSClock::now();
        comp_running_ = true;
    }

    // Ket thuc do thoi gian tinh toan.
    void stop_comp() {
        if (!comp_running_) return;
        comp_sec += elapsed_seconds(comp_start_, TTSSClock::now());
        comp_running_ = false;
    }

    // Bat dau do thoi gian giao tiep.
    void start_comm() {
        if (comm_running_) return;
        comm_start_ = TTSSClock::now();
        comm_running_ = true;
    }

    // Ket thuc do thoi gian giao tiep.
    void stop_comm() {
        if (!comm_running_) return;
        comm_sec += elapsed_seconds(comm_start_, TTSSClock::now());
        comm_running_ = false;
    }
};

// Lay gia tri lon nhat cua tung pha qua moi tien trinh trong comm.
// Moi tien trinh deu nhan ket qua. Dung ham nay truoc khi ghi output.csv.
inline PhaseTimer reduce_max(const PhaseTimer& t, MPI_Comm comm = MPI_COMM_WORLD) {
    double in[4] = {t.init_sec, t.comp_sec, t.comm_sec, t.total_sec};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(in, out, 4, MPI_DOUBLE, MPI_MAX, comm);

    PhaseTimer r;
    r.init_sec = out[0];
    r.comp_sec = out[1];
    r.comm_sec = out[2];
    r.total_sec = out[3];
    return r;
}