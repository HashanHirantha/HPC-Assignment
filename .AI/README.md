# .AI — Conway's Game of Life (HPC Assignment)

## Overview
This directory contains a complete implementation and performance comparison of **Conway's Game of Life** using two High Performance Computing approaches:

1. **CPU (2D Arrays)** — Sequential C implementation
2. **CUDA (GPU)** — Parallel NVIDIA CUDA implementation

## Files

| File | Description |
|---|---|
| `game_of_life_cpu.c` | CPU implementation using 2D arrays in C |
| `game_of_life_cuda.cu` | CUDA GPU implementation with shared memory optimization |
| `Makefile` | Build system for compiling and benchmarking |
| `report.md` | Detailed performance comparison report |
| `README.md` | This file |

## Quick Start

```bash
# Build both versions
make all

# Run CPU version (default: 1024×1024, 100 iterations)
make run_cpu

# Run CUDA version
make run_cuda

# Benchmark across multiple grid sizes
make benchmark

# Custom grid size
./game_of_life_cpu 2048 100
./game_of_life_cuda 2048 100
```

## Key Results

- **Grid Size Tested:** 128×128 to 4096×4096
- **Iterations:** 100
- **CUDA Speedup:** Up to **30×** faster for large grids
- **Crossover Point:** CUDA outperforms CPU at grid sizes ≥ 256×256
