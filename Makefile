# ============================================================================
# Makefile — Conway's Game of Life (CPU + CUDA)
# ============================================================================
#
# Targets:
#   make cpu        — Build CPU version
#   make cuda       — Build CUDA version
#   make all        — Build both
#   make run_cpu    — Build and run CPU version
#   make run_cuda   — Build and run CUDA version
#   make benchmark  — Run both versions across multiple grid sizes
#   make clean      — Remove compiled binaries
# ============================================================================

CC       = gcc
NVCC     = nvcc
CFLAGS   = -O2 -Wall
NVFLAGS  = -O2

# Default grid size and iterations
GRID_SIZE   = 1024
ITERATIONS  = 100

# Grid sizes for benchmarking
BENCH_SIZES = 128 256 512 1024 2048 4096

.PHONY: all cpu cuda run_cpu run_cuda benchmark clean

all: cpu cuda

# ── CPU Build ────────────────────────────────────────────────────────────
cpu: game_of_life_cpu

game_of_life_cpu: game_of_life_cpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# ── CUDA Build ───────────────────────────────────────────────────────────
cuda: game_of_life_cuda

game_of_life_cuda: game_of_life_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

# ── Run Targets ──────────────────────────────────────────────────────────
run_cpu: cpu
	./game_of_life_cpu $(GRID_SIZE) $(ITERATIONS)

run_cuda: cuda
	./game_of_life_cuda $(GRID_SIZE) $(ITERATIONS)

# ── Benchmark across multiple grid sizes ─────────────────────────────────
benchmark: cpu cuda
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║         BENCHMARK: CPU vs CUDA across Grid Sizes           ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@for size in $(BENCH_SIZES); do \
		echo "━━━━━━━━━━━━━━━━━ Grid Size: $$size x $$size ━━━━━━━━━━━━━━━━━"; \
		echo ""; \
		./game_of_life_cpu $$size $(ITERATIONS); \
		echo ""; \
		./game_of_life_cuda $$size $(ITERATIONS); \
		echo ""; \
	done

# ── Cleanup ──────────────────────────────────────────────────────────────
clean:
	rm -f game_of_life_cpu game_of_life_cuda game_of_life_cpu.exe game_of_life_cuda.exe
