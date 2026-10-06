# ============================================================================
# Makefile — Conway's Game of Life (CPU + CUDA)
# ============================================================================
#
# Targets:
#   make cpu          — Build CPU version
#   make cuda         — Build CUDA version
#   make terminal     — Build terminal live interactive version
#   make all          — Build both CPU and CUDA
#   make run_cpu      — Build and run CPU version
#   make run_cuda     — Build and run CUDA version
#   make run_terminal — Build and run live terminal animated version
#   make live         — Open live web interactive simulator in browser
#   make benchmark    — Run both versions across multiple grid sizes
#   make clean        — Remove compiled binaries
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

.PHONY: all cpu cuda terminal run_cpu run_cuda run_terminal live benchmark clean

all: cpu cuda

# ── CPU Build ────────────────────────────────────────────────────────────
cpu: game_of_life_cpu

game_of_life_cpu: game_of_life_cpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# ── Terminal Live Build ──────────────────────────────────────────────────
terminal:
	$(CC) $(CFLAGS) -o game_of_life_terminal game_of_life_terminal.c

# ── CUDA Build ───────────────────────────────────────────────────────────
cuda: game_of_life_cuda

game_of_life_cuda: game_of_life_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

# ── Run Targets ──────────────────────────────────────────────────────────
run_cpu: cpu
	./game_of_life_cpu $(GRID_SIZE) $(ITERATIONS)

run_cuda: cuda
	@echo "Checking for physical NVIDIA GPU hardware..."
	./game_of_life_cuda $(GRID_SIZE) $(ITERATIONS)

run_terminal:
	@if not exist game_of_life_terminal.exe $(CC) $(CFLAGS) -o game_of_life_terminal game_of_life_terminal.c 2>nul || true
	./game_of_life_terminal 64 32 50 42

live:
	explorer.exe index.html || start index.html || xdg-open index.html

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
	rm -f game_of_life_cpu game_of_life_cuda game_of_life_terminal game_of_life_cpu.exe game_of_life_cuda.exe game_of_life_terminal.exe
