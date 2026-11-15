#ifndef CANNON_GRID_H
#define CANNON_GRID_H

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cannon {

// Cau truc luoi tuan hoan q x q (voi p = q * q)
struct CannonGrid {
    MPI_Comm comm      = MPI_COMM_NULL; 
    int p              = 1;             // Tong so tien trinh
    int q              = 1;             // Kich thuoc luoi moi chieu: q = sqrt(p)
    int rank           = 0;             // Rank trong comm Cartesian
    int row            = 0;             // Toa do hang cua tien trinh (0 .. q-1)
    int col            = 0;             // Toa do cot cua tien trinh (0 .. q-1)

    int left           = MPI_PROC_NULL;
    int right          = MPI_PROC_NULL;
    int up             = MPI_PROC_NULL;
    int down           = MPI_PROC_NULL;

    int align_a_src    = MPI_PROC_NULL;
    int align_a_dst    = MPI_PROC_NULL;
    int align_b_src    = MPI_PROC_NULL;
    int align_b_dst    = MPI_PROC_NULL;

    int n              = 0;
    int n_pad          = 0;
    int b              = 0;
};

inline int grid_dim_from_procs(int p) {
    if (p <= 0) return -1;
    int q = static_cast<int>(std::round(std::sqrt(static_cast<double>(p))));
    return (q * q == p) ? q : -1;
}

inline int compute_padded_size(int n, int q) {
    if (n <= 0 || q <= 0) return 0;
    int rem = n % q;
    return (rem == 0) ? n : (n + (q - rem));
}

inline bool init_grid(CannonGrid& g, int n, MPI_Comm parent_comm = MPI_COMM_WORLD) {
    MPI_Comm_size(parent_comm, &g.p);
    g.q = grid_dim_from_procs(g.p);
    if (g.q <= 0) {
        int world_rank = 0;
        MPI_Comm_rank(parent_comm, &world_rank);
        if (world_rank == 0) {
            std::cerr << "[ERROR] So tien trinh p=" << g.p
                      << " khong phai so chinh phuong (p = q^2)!\n";
        }
        return false;
    }

    g.n     = n;
    g.n_pad = compute_padded_size(n, g.q);
    g.b     = g.n_pad / g.q;

    int dims[2]    = {g.q, g.q};
    int periods[2] = {1, 1}; // Luoi tuan hoan (periodic torus)
    MPI_Cart_create(parent_comm, 2, dims, periods, 0, &g.comm);

    MPI_Comm_rank(g.comm, &g.rank);
    int coords[2] = {0, 0};
    MPI_Cart_coords(g.comm, g.rank, 2, coords);
    g.row = coords[0];
    g.col = coords[1];

    MPI_Cart_shift(g.comm, 1, 1, &g.left, &g.right);
    MPI_Cart_shift(g.comm, 0, 1, &g.up,   &g.down);

    MPI_Cart_shift(g.comm, 1, -g.row, &g.align_a_src, &g.align_a_dst);
    MPI_Cart_shift(g.comm, 0, -g.col, &g.align_b_src, &g.align_b_dst);

    return true;
}

inline void free_grid(CannonGrid& g) {
    if (g.comm != MPI_COMM_NULL && g.comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&g.comm);
        g.comm = MPI_COMM_NULL;
    }
}

inline void pad_matrix(const std::vector<double>& src,
                       std::vector<double>& dst_padded,
                       int n,
                       int n_pad) {
    dst_padded.assign(static_cast<size_t>(n_pad) * n_pad, 0.0);
    for (int i = 0; i < n; ++i) {
        std::copy_n(src.data() + static_cast<size_t>(i) * n,
                    n,
                    dst_padded.data() + static_cast<size_t>(i) * n_pad);
    }
}

inline void unpad_matrix(const std::vector<double>& src_padded,
                         std::vector<double>& dst,
                         int n,
                         int n_pad) {
    dst.resize(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        std::copy_n(src_padded.data() + static_cast<size_t>(i) * n_pad,
                    n,
                    dst.data() + static_cast<size_t>(i) * n);
    }
}

inline void scatter_matrix(const CannonGrid& g,
                           const std::vector<double>& global_mat,
                           std::vector<double>& local_block) {
    const int b          = g.b;
    const int q          = g.q;
    const int n          = g.n;
    const int n_pad      = g.n_pad;
    const int block_elems = b * b;

    local_block.assign(static_cast<size_t>(block_elems), 0.0);
    std::vector<double> packed_send;

    if (g.rank == 0) {
        std::vector<double> padded;
        const double* mat_ptr = nullptr;
        if (n == n_pad) {
            mat_ptr = global_mat.data();
        } else {
            pad_matrix(global_mat, padded, n, n_pad);
            mat_ptr = padded.data();
        }

        packed_send.resize(static_cast<size_t>(n_pad) * n_pad, 0.0);
        for (int proc = 0; proc < g.p; ++proc) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(g.comm, proc, 2, coords);
            int pr = coords[0];
            int pc = coords[1];

            double* dst_block = packed_send.data() + static_cast<size_t>(proc) * block_elems;
            for (int i = 0; i < b; ++i) {
                const double* src_row = mat_ptr
                    + static_cast<size_t>(pr * b + i) * n_pad
                    + static_cast<size_t>(pc * b);
                std::copy_n(src_row, b, dst_block + static_cast<size_t>(i) * b);
            }
        }
    }

    MPI_Scatter(g.rank == 0 ? packed_send.data() : nullptr,
                block_elems,
                MPI_DOUBLE,
                local_block.data(),
                block_elems,
                MPI_DOUBLE,
                0,
                g.comm);
}

inline void gather_matrix(const CannonGrid& g,
                          const std::vector<double>& local_block,
                          std::vector<double>& global_mat) {
    const int b           = g.b;
    const int n           = g.n;
    const int n_pad       = g.n_pad;
    const int block_elems = b * b;

    std::vector<double> packed_recv;
    if (g.rank == 0) {
        packed_recv.resize(static_cast<size_t>(n_pad) * n_pad, 0.0);
    }

    MPI_Gather(local_block.data(),
               block_elems,
               MPI_DOUBLE,
               g.rank == 0 ? packed_recv.data() : nullptr,
               block_elems,
               MPI_DOUBLE,
               0,
               g.comm);

    if (g.rank == 0) {
        std::vector<double> padded(static_cast<size_t>(n_pad) * n_pad, 0.0);
        for (int proc = 0; proc < g.p; ++proc) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(g.comm, proc, 2, coords);
            int pr = coords[0];
            int pc = coords[1];

            const double* src_block = packed_recv.data() + static_cast<size_t>(proc) * block_elems;
            for (int i = 0; i < b; ++i) {
                double* dst_row = padded.data()
                    + static_cast<size_t>(pr * b + i) * n_pad
                    + static_cast<size_t>(pc * b);
                std::copy_n(src_block + static_cast<size_t>(i) * b, b, dst_row);
            }
        }

        if (n == n_pad) {
            global_mat = std::move(padded);
        } else {
            unpad_matrix(padded, global_mat, n, n_pad);
        }
    }
}

inline void initial_align(const CannonGrid& g,
                          std::vector<double>& local_A,
                          std::vector<double>& local_B) {
    const int count = g.b * g.b;
    MPI_Sendrecv_replace(local_A.data(),
                         count,
                         MPI_DOUBLE,
                         g.align_a_dst,
                         101,
                         g.align_a_src,
                         101,
                         g.comm,
                         MPI_STATUS_IGNORE);

    MPI_Sendrecv_replace(local_B.data(),
                         count,
                         MPI_DOUBLE,
                         g.align_b_dst,
                         102,
                         g.align_b_src,
                         102,
                         g.comm,
                         MPI_STATUS_IGNORE);
}

inline void shift_blocks_once(const CannonGrid& g,
                              std::vector<double>& local_A,
                              std::vector<double>& local_B) {
    const int count = g.b * g.b;
    MPI_Sendrecv_replace(local_A.data(),
                         count,
                         MPI_DOUBLE,
                         g.left,
                         201,
                         g.right,
                         201,
                         g.comm,
                         MPI_STATUS_IGNORE);

    MPI_Sendrecv_replace(local_B.data(),
                         count,
                         MPI_DOUBLE,
                         g.up,
                         202,
                         g.down,
                         202,
                         g.comm,
                         MPI_STATUS_IGNORE);
}

inline void block_mul_add(const double* A,
                          const double* B,
                          double* C,
                          int b,
                          int block_size = 0) {
    if (block_size <= 0 || block_size >= b) {
        for (int i = 0; i < b; ++i) {
            for (int k = 0; k < b; ++k) {
                const double a_ik = A[i * b + k];
                for (int j = 0; j < b; ++j) {
                    C[i * b + j] += a_ik * B[k * b + j];
                }
            }
        }
    } else {
        const int bs = block_size;
        for (int ii = 0; ii < b; ii += bs) {
            const int i_end = std::min(ii + bs, b);
            for (int kk = 0; kk < b; kk += bs) {
                const int k_end = std::min(kk + bs, b);
                for (int jj = 0; jj < b; jj += bs) {
                    const int j_end = std::min(jj + bs, b);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            const double a_ik = A[i * b + k];
                            for (int j = jj; j < j_end; ++j) {
                                C[i * b + j] += a_ik * B[k * b + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void block_mul_add(const std::vector<double>& A,
                          const std::vector<double>& B,
                          std::vector<double>& C,
                          int b,
                          int block_size = 0) {
    block_mul_add(A.data(), B.data(), C.data(), b, block_size);
}

inline void print_distributed_blocks(const CannonGrid& g,
                                     const std::vector<double>& local_block,
                                     const std::string& label) {
    for (int r = 0; r < g.p; ++r) {
        MPI_Barrier(g.comm);
        if (g.rank == r) {
            std::ostringstream oss;
            oss << "--- [" << label << "] Rank " << g.rank
                << " | Grid(" << g.row << "," << g.col << ")"
                << " | Neighbors(L=" << g.left << ", R=" << g.right
                << ", U=" << g.up << ", D=" << g.down << ") ---\n";
            for (int i = 0; i < g.b; ++i) {
                for (int j = 0; j < g.b; ++j) {
                    oss << std::setw(6) << std::fixed << std::setprecision(1)
                        << local_block[i * g.b + j] << " ";
                }
                oss << "\n";
            }
            std::cout << oss.str() << std::flush;
        }
    }
    MPI_Barrier(g.comm);
}

} // namespace cannon

#endif // CANNON_GRID_H