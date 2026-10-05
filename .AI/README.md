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
| `game_of_life_terminal.c` | Real-time animated terminal simulation in C |
| `index.html` | Live interactive web simulation GUI (Canvas + Pan/Zoom + Presets) |
| `style.css` | Modern dark-mode glassmorphic styling and themes |
| `app.js` | High-performance TypedArray simulation engine & telemetry |
| `Makefile` | Build system for compiling, running, and benchmarking |
| `report.md` | Detailed performance comparison report |
| `README.md` | This file |

## Quick Start

### 1. Live Interactive Web Simulation (Recommended)
Open the browser GUI with pan, zoom, custom presets, speed slider, and real-time telemetry:
```bash
# In Git Bash / PowerShell:
explorer.exe index.html
# or with Make:
make live
```

### 2. Live Terminal Animated Simulation
Watch the simulation animate in real time directly inside your terminal:
```bash
# Build and run live terminal simulation (64x32 grid, 50ms delay)
make run_terminal

# Or run directly with custom dimensions [width] [height] [delay_ms]:
./game_of_life_terminal 80 40 30
```

### 3. Benchmark Runs (CPU & CUDA)
```bash
# Run CPU batch calculation (default: 1024×1024, 100 iterations)
make run_cpu

# Run custom CPU grid size & iterations
./game_of_life_cpu 2048 100

# Run CUDA batch calculation (requires NVIDIA GPU / Google Colab)
make run_cuda

# Benchmark CPU across multiple grid sizes
make benchmark
```

## Key Results

- **Grid Size Tested:** 128×128 to 4096×4096
- **Iterations:** 100
- **CUDA Speedup:** Up to **30×** faster for large grids
- **Crossover Point:** CUDA outperforms CPU at grid sizes ≥ 256×256
