/*
 * ============================================================================
 * Conway's Game of Life — CPU Implementation (2D Arrays)
 * ============================================================================
 * 
 * This program simulates Conway's Game of Life on a 2D grid using standard
 * C arrays. It runs for a configurable number of iterations and measures
 * execution time using high-resolution timers.
 *
 * Rules of Conway's Game of Life:
 *   1. Any live cell with fewer than 2 live neighbours dies (underpopulation).
 *   2. Any live cell with 2 or 3 live neighbours lives on.
 *   3. Any live cell with more than 3 live neighbours dies (overpopulation).
 *   4. Any dead cell with exactly 3 live neighbours becomes alive (reproduction).
 *
 * Compilation:
 *   gcc -O2 -o game_of_life_cpu game_of_life_cpu.c -lm
 *
 * Usage:
 *   ./game_of_life_cpu [grid_size] [num_iterations]
 *   Default: grid_size = 1024, num_iterations = 100
 *
 * Author:  HPC Assignment
 * Date:    October 2026
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/* ── High-Resolution Timer ─────────────────────────────────────────────── */

double get_time_in_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

/* ── Grid Allocation & Deallocation ────────────────────────────────────── */

/*
 * Allocates a 2D grid of size N x N as a contiguous block of memory.
 * Returns a pointer to an array of row pointers for grid[row][col] access.
 */
int **allocate_grid(int N) {
    int **grid = (int **)malloc(N * sizeof(int *));
    int *data  = (int *)calloc(N * N, sizeof(int));
    if (!grid || !data) {
        fprintf(stderr, "Error: Memory allocation failed for grid of size %d x %d\n", N, N);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; i++) {
        grid[i] = data + i * N;
    }
    return grid;
}

/*
 * Frees the memory associated with a grid allocated by allocate_grid().
 */
void free_grid(int **grid) {
    if (grid) {
        free(grid[0]);  /* Free the contiguous data block */
        free(grid);     /* Free the row-pointer array     */
    }
}

/* ── Grid Initialization ──────────────────────────────────────────────── */

/*
 * Initializes the grid with a random pattern.
 * Approximately 30% of cells are set to alive (1).
 * Uses a fixed seed for reproducibility across runs.
 */
void initialize_grid_random(int **grid, int N, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = (rand() % 100 < 30) ? 1 : 0;
        }
    }
}

/* ── Neighbour Counting ───────────────────────────────────────────────── */

/*
 * Counts the number of live neighbours for cell (row, col).
 * Uses toroidal (wrap-around) boundary conditions so the grid
 * behaves as if it were on the surface of a torus.
 */
static inline int count_neighbours(int **grid, int N, int row, int col) {
    int count = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;  /* Skip the cell itself */
            int ni = (row + di + N) % N;
            int nj = (col + dj + N) % N;
            count += grid[ni][nj];
        }
    }
    return count;
}

/* ── Single Step Evolution ─────────────────────────────────────────────── */

/*
 * Advances the simulation by one generation.
 * Reads from 'current' and writes to 'next'.
 */
void step(int **current, int **next, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int neighbours = count_neighbours(current, N, i, j);
            if (current[i][j] == 1) {
                /* Live cell survives with 2 or 3 neighbours */
                next[i][j] = (neighbours == 2 || neighbours == 3) ? 1 : 0;
            } else {
                /* Dead cell becomes alive with exactly 3 neighbours */
                next[i][j] = (neighbours == 3) ? 1 : 0;
            }
        }
    }
}

/* ── Population Counter ────────────────────────────────────────────────── */

/*
 * Counts the total number of live cells on the grid.
 */
long count_alive(int **grid, int N) {
    long count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            count += grid[i][j];
        }
    }
    return count;
}

/* ── Main Program ──────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    /* ── Parse command-line arguments ── */
    int N              = (argc > 1) ? atoi(argv[1]) : 1024;
    int num_iterations = (argc > 2) ? atoi(argv[2]) : 100;
    unsigned int seed  = 42;  /* Fixed seed for reproducibility */

    printf("============================================================\n");
    printf("  Conway's Game of Life — CPU (2D Array) Implementation\n");
    printf("============================================================\n");
    printf("  Grid Size       : %d x %d\n", N, N);
    printf("  Total Cells     : %d\n", N * N);
    printf("  Iterations      : %d\n", num_iterations);
    printf("  Memory per Grid : %.2f MB\n", (double)(N * N * sizeof(int)) / (1024.0 * 1024.0));
    printf("============================================================\n\n");

    /* ── Allocate grids ── */
    int **grid_current = allocate_grid(N);
    int **grid_next    = allocate_grid(N);

    /* ── Initialize ── */
    initialize_grid_random(grid_current, N, seed);
    long initial_alive = count_alive(grid_current, N);
    printf("  Initial live cells: %ld (%.2f%%)\n\n", initial_alive,
           100.0 * initial_alive / (N * N));

    /* ── Run simulation and measure time ── */
    printf("  Running %d iterations...\n", num_iterations);
    double start_time = get_time_in_seconds();

    for (int iter = 0; iter < num_iterations; iter++) {
        step(grid_current, grid_next, N);

        /* Swap grids (pointer swap, no data copy) */
        int **temp    = grid_current;
        grid_current  = grid_next;
        grid_next     = temp;
    }

    double end_time   = get_time_in_seconds();
    double total_time = end_time - start_time;

    /* ── Report results ── */
    long final_alive = count_alive(grid_current, N);
    double cells_per_second = (double)N * N * num_iterations / total_time;

    printf("\n  ──── Results ────────────────────────────────────────────\n");
    printf("  Final live cells  : %ld (%.2f%%)\n", final_alive,
           100.0 * final_alive / (N * N));
    printf("  Total time        : %.6f seconds\n", total_time);
    printf("  Avg time/iteration: %.6f seconds\n", total_time / num_iterations);
    printf("  Throughput        : %.2f million cells/second\n",
           cells_per_second / 1e6);
    printf("  ────────────────────────────────────────────────────────\n");

    /* ── Cleanup ── */
    free_grid(grid_current);
    free_grid(grid_next);

    return 0;
}
