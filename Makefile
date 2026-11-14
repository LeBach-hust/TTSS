CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Iinclude
OMPFLAGS ?= -fopenmp
BIN      := bin

.PHONY: all seq omp mpi hybrid clean
all: seq omp mpi hybrid

seq:
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) src/dot/dot_seq.cpp -o $(BIN)/dot_seq
	$(CXX) $(CXXFLAGS) src/matmul/matmul_seq.cpp -o $(BIN)/matmul_seq

omp:
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) src/dot/dot_omp.cpp -o $(BIN)/dot_omp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) src/matmul/matmul_omp.cpp -o $(BIN)/matmul_omp

mpi:
	@mkdir -p $(BIN)
	$(MPICXX) $(CXXFLAGS) src/dot/dot_mpi.cpp -o $(BIN)/dot_mpi
	$(MPICXX) $(CXXFLAGS) src/matmul/cannon_mpi.cpp -o $(BIN)/cannon_mpi
	$(MPICXX) $(CXXFLAGS) src/matmul/cannon_mpi_nb.cpp -o $(BIN)/cannon_mpi_nb

hybrid:
	@mkdir -p $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) src/dot/dot_hybrid.cpp -o $(BIN)/dot_hybrid
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) src/matmul/cannon_hybrid.cpp -o $(BIN)/cannon_hybrid
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) src/matmul/cannon_hybrid_nb.cpp -o $(BIN)/cannon_hybrid_nb

clean:
	rm -rf $(BIN) *.o
