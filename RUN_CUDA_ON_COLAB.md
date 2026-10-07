# How to Run CUDA on Google Colab (Free NVIDIA GPU)

> **Hardware Note for Your HPC Assignment:**  
> Your laptop is equipped with an **NVIDIA GeForce RTX 2050 GPU** (CUDA 13.1, Driver 592.82)! You can run real-time CUDA parallel simulation directly in the live interactive interface (`index.html`), compile native CUDA code locally, or use **Google Colab** (NVIDIA Tesla T4 GPU) as a cloud HPC benchmark environment.

---

## 3-Minute Quick Guide

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com](https://colab.research.google.com).
2. Click **New Notebook**.

---

### Step 2: Enable the NVIDIA GPU
1. In the top menu, click **Runtime** > **Change runtime type**.
2. Under *Hardware accelerator*, select **T4 GPU**.
3. Click **Save**.

---

### Step 3: Verify the NVIDIA GPU
In the first notebook cell, type and run:
```bash
!nvidia-smi
```
You will see output showing your active NVIDIA GPU (e.g., `Tesla T4`, `15360 MiB VRAM`, CUDA Version 12.x).

---

### Step 4: Upload your CUDA Source Code
Run this cell in Colab to upload `game_of_life_cuda.cu` from your computer:
```python
from google.colab import files
uploaded = files.upload()
```
Select `game_of_life_cuda.cu` from `e:\HPC Assignment\game_of_life_cuda.cu`.

---

### Step 5: Compile and Run on Real NVIDIA GPU
Run this cell to compile with NVIDIA NVCC:
```bash
!nvcc -O2 -o game_of_life_cuda game_of_life_cuda.cu
!./game_of_life_cuda 1024 100
```

You will see output like this:
```
============================================================
  Conway's Game of Life — CUDA (GPU) Implementation
============================================================
  Grid Size       : 1024 x 1024
  Total Cells     : 1048576
  Iterations      : 100
  Block Size      : 16 x 16 (256 threads/block)
  Grid Dimension  : 64 x 64 blocks
============================================================
  Total GPU Time  : 0.285 seconds
  Avg Time / Iter : 0.00285 seconds
  Throughput      : 367.92 million cells/second
============================================================
```

---

### Step 6: Run Full Benchmark (All Grid Sizes)
To get the exact benchmark data for your assignment report, run:
```bash
for size in 128 256 512 1024 2048 4096; do
  echo "--- Testing Grid Size: ${size}x${size} ---"
  ./game_of_life_cuda $size 100
done
```

Take screenshots of this output to include in your assignment report!
