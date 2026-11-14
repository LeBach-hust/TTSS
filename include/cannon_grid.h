#pragma once

// Shared Cartesian-grid, padding, block/tile helpers for Cannon implementations.
// TODO: add MPI Cartesian communicator setup and matrix block operations.
namespace ttss {
struct CannonGrid {
    int rows = 0;
    int cols = 0;
    int rank = 0;
    int size = 0;
};
}  // namespace ttss
