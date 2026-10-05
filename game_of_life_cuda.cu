/*
 * ============================================================================
 * Conway's Game of Life — CUDA (GPU) Implementation
 * ============================================================================
 *
 * This program simulates Conway's Game of Life on a 2D grid using NVIDIA
 * CUDA for massively parallel execution on the GPU. Each cell is processed
 * by a separate CUDA thread, and shared memory is used to reduce global
 * memory accesses.
 *
 * Rules of Conway's Game of Life:
 *   1. Any live cell with fewer than 2 live neighbours dies (underpopulation).
 *   2. Any live cell with 2 or 3 live neighbours lives on.
 *   3. Any live cell with more than 3 live neighbours dies (overpopulation).
 *   4. Any dead cell with exactly 3 live neighbours becomes alive (reproduction).
 *
 * Compilation:
 *   nvcc -O2 -o game_of_life_cuda game_of_life_cuda.cu
 *
 * Usage:
 *   ./game_of_life_cuda [grid_size] [num_iterations]
 *   Default: grid_size = 1024, num_iterations = 100
 *
 * Author:  HPC Assignment
 * Date:    October 2026
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* ── CUDA Error Checking Macro ─────────────────────────────────────────── */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ── CUDA Thread Block Configuration ──────────────────────────────────── */

#define BLOCK_SIZE 16   /* Threads per block dimension (16x16 = 256 threads) */

/* ── Shared Memory Tile Size ──────────────────────────────────────────── */
/* Each block loads a (BLOCK_SIZE+2) x (BLOCK_SIZE+2) tile into shared
 * memory, which includes a 1-cell halo border for neighbour lookups. */
#define TILE_SIZE (BLOCK_SIZE + 2)

/* ── CUDA Kernel: Game of Life Step (Shared Memory Optimized) ─────────── */

/*
 * Each thread block loads a tile of cells (including halo border) into
 * shared memory. Interior threads then compute the next state using
 * fast shared memory reads instead of slow global memory reads.
 *
 * Toroidal (wrap-around) boundary conditions are applied.
 */
__global__ void game_of_life_kernel(const int *current, int *next, int N) {
    /* ── Shared memory tile with halo ── */
    __shared__ int tile[TILE_SIZE][TILE_SIZE];

    /* ── Global coordinates of this thread ── */
    int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int gy = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    /* ── Local coordinates within the tile (offset by 1 for halo) ── */
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    /* ── Load interior cell into shared memory ── */
    if (gx < N && gy < N) {
        tile[ly][lx] = current[gy * N + gx];
    } else {
        tile[ly][lx] = 0;
    }

    /* ── Load halo cells (borders of the tile) ── */

    /* Top halo row */
    if (threadIdx.y == 0) {
        int halo_y = (gy - 1 + N) % N;
        tile[0][lx] = (gx < N) ? current[halo_y * N + gx] : 0;
    }
    /* Bottom halo row */
    if (threadIdx.y == BLOCK_SIZE - 1 || gy == N - 1) {
        int halo_y = (gy + 1) % N;
        tile[ly + 1][lx] = (gx < N) ? current[halo_y * N + gx] : 0;
    }
    /* Left halo column */
    if (threadIdx.x == 0) {
        int halo_x = (gx - 1 + N) % N;
        tile[ly][0] = (gy < N) ? current[gy * N + halo_x] : 0;
    }
    /* Right halo column */
    if (threadIdx.x == BLOCK_SIZE - 1 || gx == N - 1) {
        int halo_x = (gx + 1) % N;
        tile[ly][lx + 1] = (gy < N) ? current[gy * N + halo_x] : 0;
    }

    /* ── Load halo corners ── */
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[0][0] = current[((gy - 1 + N) % N) * N + ((gx - 1 + N) % N)];
    }
    if ((threadIdx.x == BLOCK_SIZE - 1 || gx == N - 1) && threadIdx.y == 0) {
        tile[0][lx + 1] = current[((gy - 1 + N) % N) * N + ((gx + 1) % N)];
    }
    if (threadIdx.x == 0 && (threadIdx.y == BLOCK_SIZE - 1 || gy == N - 1)) {
        tile[ly + 1][0] = current[((gy + 1) % N) * N + ((gx - 1 + N) % N)];
    }
    if ((threadIdx.x == BLOCK_SIZE - 1 || gx == N - 1) &&
        (threadIdx.y == BLOCK_SIZE - 1 || gy == N - 1)) {
        tile[ly + 1][lx + 1] = current[((gy + 1) % N) * N + ((gx + 1) % N)];
    }

    /* ── Synchronize to ensure all shared memory is loaded ── */
    __syncthreads();

    /* ── Compute next state using shared memory ── */
    if (gx < N && gy < N) {
        int neighbours = tile[ly - 1][lx - 1] + tile[ly - 1][lx] + tile[ly - 1][lx + 1]
                       + tile[ly    ][lx - 1]                     + tile[ly    ][lx + 1]
                       + tile[ly + 1][lx - 1] + tile[ly + 1][lx] + tile[ly + 1][lx + 1];

        int cell = tile[ly][lx];
        if (cell == 1) {
            next[gy * N + gx] = (neighbours == 2 || neighbours == 3) ? 1 : 0;
        } else {
            next[gy * N + gx] = (neighbours == 3) ? 1 : 0;
        }
    }
}

/* ── Simple Kernel (No Shared Memory — for comparison) ─────────────────── */

__global__ void game_of_life_kernel_simple(const int *current, int *next, int N) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= N || gy >= N) return;

    int neighbours = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            int ni = (gy + di + N) % N;
            int nj = (gx + dj + N) % N;
            neighbours += current[ni * N + nj];
        }
    }

    int cell = current[gy * N + gx];
    if (cell == 1) {
        next[gy * N + gx] = (neighbours == 2 || neighbours == 3) ? 1 : 0;
    } else {
        next[gy * N + gx] = (neighbours == 3) ? 1 : 0;
    }
}

/* ── Host-Side Grid Initialization ─────────────────────────────────────── */

void initialize_grid_random(int *grid, int N, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < N * N; i++) {
        grid[i] = (rand() % 100 < 30) ? 1 : 0;
    }
}

/* ── Population Counter ────────────────────────────────────────────────── */

long count_alive(int *grid, int N) {
    long count = 0;
    for (int i = 0; i < N * N; i++) {
        count += grid[i];
    }
    return count;
}

/* ── Print GPU Device Info ─────────────────────────────────────────────── */

void print_device_info(void) {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("  GPU Device        : %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM Count          : %d\n", prop.multiProcessorCount);
    printf("  Global Memory     : %.2f GB\n",
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Mem/Block  : %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Max Threads/Block : %d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size         : %d\n", prop.warpSize);
}

/* ── Main Program ──────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    /* ── Parse command-line arguments ── */
    int N              = (argc > 1) ? atoi(argv[1]) : 1024;
    int num_iterations = (argc > 2) ? atoi(argv[2]) : 100;
    unsigned int seed  = 42;  /* Fixed seed for reproducibility */

    size_t grid_bytes = (size_t)N * N * sizeof(int);

    printf("============================================================\n");
    printf("  Conway's Game of Life — CUDA (GPU) Implementation\n");
    printf("============================================================\n");
    printf("  Grid Size         : %d x %d\n", N, N);
    printf("  Total Cells       : %d\n", N * N);
    printf("  Iterations        : %d\n", num_iterations);
    printf("  Block Size        : %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("  Memory per Grid   : %.2f MB\n", (double)grid_bytes / (1024.0 * 1024.0));
    printf("------------------------------------------------------------\n");
    print_device_info();
    printf("============================================================\n\n");

    /* ── Allocate host memory ── */
    int *h_grid = (int *)malloc(grid_bytes);
    if (!h_grid) {
        fprintf(stderr, "Error: Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    /* ── Initialize grid on host ── */
    initialize_grid_random(h_grid, N, seed);
    long initial_alive = count_alive(h_grid, N);
    printf("  Initial live cells: %ld (%.2f%%)\n\n", initial_alive,
           100.0 * initial_alive / ((double)N * N));

    /* ── Allocate device memory ── */
    int *d_current, *d_next;
    CUDA_CHECK(cudaMalloc((void **)&d_current, grid_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_next, grid_bytes));

    /* ── Copy initial grid to device ── */
    CUDA_CHECK(cudaMemcpy(d_current, h_grid, grid_bytes, cudaMemcpyHostToDevice));

    /* ── Configure kernel launch parameters ── */
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("  Grid dimensions   : (%d, %d) blocks\n", gridDim.x, gridDim.y);
    printf("  Block dimensions  : (%d, %d) threads\n", blockDim.x, blockDim.y);
    printf("  Total GPU threads : %d\n\n", gridDim.x * gridDim.y * BLOCK_SIZE * BLOCK_SIZE);

    /* ── Create CUDA events for timing ── */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* ══════════════════════════════════════════════════════════════════
     *  Run: Shared Memory Optimized Kernel
     * ══════════════════════════════════════════════════════════════════ */
    printf("  Running %d iterations (Shared Memory Kernel)...\n", num_iterations);

    /* Reset grid */
    CUDA_CHECK(cudaMemcpy(d_current, h_grid, grid_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < num_iterations; iter++) {
        game_of_life_kernel<<<gridDim, blockDim>>>(d_current, d_next, N);
        /* Swap device pointers */
        int *temp  = d_current;
        d_current  = d_next;
        d_next     = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float shared_mem_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&shared_mem_time_ms, start, stop));
    double shared_mem_time = shared_mem_time_ms / 1000.0;

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(h_grid, d_current, grid_bytes, cudaMemcpyDeviceToHost));
    long final_alive_shared = count_alive(h_grid, N);
    double throughput_shared = (double)N * N * num_iterations / shared_mem_time;

    printf("\n  ──── Results (Shared Memory Kernel) ─────────────────────\n");
    printf("  Final live cells  : %ld (%.2f%%)\n", final_alive_shared,
           100.0 * final_alive_shared / ((double)N * N));
    printf("  Total time        : %.6f seconds\n", shared_mem_time);
    printf("  Avg time/iteration: %.6f seconds\n", shared_mem_time / num_iterations);
    printf("  Throughput        : %.2f million cells/second\n",
           throughput_shared / 1e6);
    printf("  ────────────────────────────────────────────────────────\n\n");

    /* ══════════════════════════════════════════════════════════════════
     *  Run: Simple (Global Memory Only) Kernel
     * ══════════════════════════════════════════════════════════════════ */

    /* Re-initialize */
    initialize_grid_random(h_grid, N, seed);
    CUDA_CHECK(cudaMemcpy(d_current, h_grid, grid_bytes, cudaMemcpyHostToDevice));

    printf("  Running %d iterations (Simple/Global Memory Kernel)...\n", num_iterations);

    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < num_iterations; iter++) {
        game_of_life_kernel_simple<<<gridDim, blockDim>>>(d_current, d_next, N);
        int *temp  = d_current;
        d_current  = d_next;
        d_next     = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float simple_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&simple_time_ms, start, stop));
    double simple_time = simple_time_ms / 1000.0;

    CUDA_CHECK(cudaMemcpy(h_grid, d_current, grid_bytes, cudaMemcpyDeviceToHost));
    long final_alive_simple = count_alive(h_grid, N);
    double throughput_simple = (double)N * N * num_iterations / simple_time;

    printf("\n  ──── Results (Simple/Global Memory Kernel) ──────────────\n");
    printf("  Final live cells  : %ld (%.2f%%)\n", final_alive_simple,
           100.0 * final_alive_simple / ((double)N * N));
    printf("  Total time        : %.6f seconds\n", simple_time);
    printf("  Avg time/iteration: %.6f seconds\n", simple_time / num_iterations);
    printf("  Throughput        : %.2f million cells/second\n",
           throughput_simple / 1e6);
    printf("  ────────────────────────────────────────────────────────\n\n");

    /* ══════════════════════════════════════════════════════════════════
     *  Summary Comparison
     * ══════════════════════════════════════════════════════════════════ */
    printf("  ╔════════════════════════════════════════════════════════╗\n");
    printf("  ║             CUDA Kernel Comparison Summary            ║\n");
    printf("  ╠══════════════════════╦═══════════════╦════════════════╣\n");
    printf("  ║ Metric               ║ Shared Memory ║ Global Memory  ║\n");
    printf("  ╠══════════════════════╬═══════════════╬════════════════╣\n");
    printf("  ║ Total Time (s)       ║ %13.6f ║ %14.6f ║\n", shared_mem_time, simple_time);
    printf("  ║ Avg Iter Time (s)    ║ %13.6f ║ %14.6f ║\n",
           shared_mem_time / num_iterations, simple_time / num_iterations);
    printf("  ║ Throughput (Mcells/s) ║ %13.2f ║ %14.2f ║\n",
           throughput_shared / 1e6, throughput_simple / 1e6);
    printf("  ║ Speedup vs Global    ║ %12.2fx ║           1.00x ║\n",
           simple_time / shared_mem_time);
    printf("  ╚══════════════════════╩═══════════════╩════════════════╝\n");

    /* ── Cleanup ── */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    free(h_grid);

    return 0;
}
