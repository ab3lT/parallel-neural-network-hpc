# Makefile for Parallel Deep Learning Training
# Distributed Computing for AI - Masters Assignment
#
# Builds: serial, OpenMP, MPI, and hybrid implementations

# Compilers
CC = gcc
MPICC = mpicc

# Flags
CFLAGS = -Wall -O3 -std=c99
OPENMP_FLAGS = -fopenmp
MPI_FLAGS = 
LDFLAGS = -lm

# Include path
INCLUDES = -I./include

# Source files
UTILS_SRC = src/utils.c
SERIAL_SRC = src/serial_train.c
OPENMP_SRC = src/openmp_train.c
MPI_SRC = src/mpi_train.c
HYBRID_SRC = src/hybrid_train.c

# Output binaries
SERIAL_BIN = bin/serial_train
OPENMP_BIN = bin/openmp_train
MPI_BIN = bin/mpi_train
HYBRID_BIN = bin/hybrid_train

# Default target
all: directories serial openmp mpi hybrid

# Create output directory
directories:
	@mkdir -p bin

# Serial implementation
serial: $(SERIAL_SRC) $(UTILS_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(SERIAL_BIN) $(SERIAL_SRC) $(UTILS_SRC) $(LDFLAGS)
	@echo "Built: $(SERIAL_BIN)"

# OpenMP implementation
openmp: $(OPENMP_SRC) $(UTILS_SRC)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $(INCLUDES) -o $(OPENMP_BIN) $(OPENMP_SRC) $(UTILS_SRC) $(LDFLAGS)
	@echo "Built: $(OPENMP_BIN)"

# MPI implementation
mpi: $(MPI_SRC) $(UTILS_SRC)
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $(MPI_BIN) $(MPI_SRC) $(UTILS_SRC) $(LDFLAGS)
	@echo "Built: $(MPI_BIN)"

# Hybrid MPI+OpenMP implementation
hybrid: $(HYBRID_SRC) $(UTILS_SRC)
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(INCLUDES) -o $(HYBRID_BIN) $(HYBRID_SRC) $(UTILS_SRC) $(LDFLAGS)
	@echo "Built: $(HYBRID_BIN)"

# Individual build targets
serial-only: directories serial
openmp-only: directories openmp
mpi-only: directories mpi
hybrid-only: directories hybrid

# Build without MPI (if MPI is not available)
no-mpi: directories serial openmp
	@echo "Built serial and OpenMP versions (no MPI)"

# Run targets with default parameters
run-serial: serial
	./$(SERIAL_BIN) -n 10000 -e 20 -b 64

run-openmp: openmp
	./$(OPENMP_BIN) -n 10000 -e 20 -b 64 -t 4

run-mpi: mpi
	mpirun -np 4 ./$(MPI_BIN) -n 10000 -e 20 -b 64

run-hybrid: hybrid
	mpirun -np 2 ./$(HYBRID_BIN) -n 10000 -e 20 -b 64 -t 2

# Run experiments for performance comparison
experiment: all
	@echo "Running performance experiments..."
	@echo ""
	@echo "=== Serial Baseline ==="
	./$(SERIAL_BIN) -n 20000 -e 10 -b 64
	@echo ""
	@echo "=== OpenMP 2 threads ==="
	./$(OPENMP_BIN) -n 20000 -e 10 -b 64 -t 2
	@echo ""
	@echo "=== OpenMP 4 threads ==="
	./$(OPENMP_BIN) -n 20000 -e 10 -b 64 -t 4
	@echo ""
	@echo "=== MPI 2 processes ==="
	mpirun -np 2 ./$(MPI_BIN) -n 20000 -e 10 -b 64
	@echo ""
	@echo "=== MPI 4 processes ==="
	mpirun -np 4 ./$(MPI_BIN) -n 20000 -e 10 -b 64
	@echo ""
	@echo "=== Hybrid 2 procs x 2 threads ==="
	mpirun -np 2 ./$(HYBRID_BIN) -n 20000 -e 10 -b 64 -t 2

# Scalability experiments (vary workload)
scalability: all
	@echo "Running scalability experiments..."
	@for n in 5000 10000 20000 40000; do \
		echo ""; \
		echo "=== Dataset size: $$n ==="; \
		echo "Serial:"; \
		./$(SERIAL_BIN) -n $$n -e 5 -b 64; \
		echo "OpenMP (4 threads):"; \
		./$(OPENMP_BIN) -n $$n -e 5 -b 64 -t 4; \
	done

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f *.csv
	@echo "Cleaned build artifacts"

# Clean everything including metrics
clean-all: clean
	rm -f *_metrics.csv
	@echo "Cleaned all files"

# Help target
help:
	@echo "Parallel Deep Learning Training - Makefile"
	@echo ""
	@echo "Build targets:"
	@echo "  all          - Build all implementations"
	@echo "  serial       - Build serial implementation only"
	@echo "  openmp       - Build OpenMP implementation only"
	@echo "  mpi          - Build MPI implementation only"
	@echo "  hybrid       - Build hybrid MPI+OpenMP implementation"
	@echo "  no-mpi       - Build serial and OpenMP only (if no MPI)"
	@echo ""
	@echo "Run targets:"
	@echo "  run-serial   - Run serial with default parameters"
	@echo "  run-openmp   - Run OpenMP with 4 threads"
	@echo "  run-mpi      - Run MPI with 4 processes"
	@echo "  run-hybrid   - Run hybrid with 2 procs x 2 threads"
	@echo ""
	@echo "Experiment targets:"
	@echo "  experiment   - Run performance comparison experiments"
	@echo "  scalability  - Run scalability experiments"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean        - Remove build artifacts"
	@echo "  clean-all    - Remove all generated files"
	@echo "  help         - Show this help message"

.PHONY: all directories serial openmp mpi hybrid \
        serial-only openmp-only mpi-only hybrid-only no-mpi \
        run-serial run-openmp run-mpi run-hybrid \
        experiment scalability clean clean-all help
