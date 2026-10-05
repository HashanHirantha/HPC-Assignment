/*
 * ============================================================================
 * Conway's Game of Life — Interactive Live Terminal Simulation
 * ============================================================================
 *
 * Real-time animated simulation with live interactive keyboard controls:
 *   [Space]      : Pause / Resume simulation
 *   [S]          : Single Step (when paused)
 *   [R]          : Randomize grid (30% density with new seed)
 *   [C]          : Clear entire grid
 *   [+] or [=]   : Speed up (decrease frame delay)
 *   [-] or [_]   : Slow down (increase frame delay)
 *   [1]          : Spawn Glider
 *   [2]          : Spawn Pulsar
 *   [3]          : Spawn Gosper Glider Gun
 *   [Q] or [ESC] : Quit simulation
 *
 * Usage:
 *   ./game_of_life_terminal [width] [height] [delay_ms] [seed]
 *   Default: width = 64, height = 32, delay_ms = 50, seed = 42
 *
 * Author:  HPC Assignment
 * Date:    October 2026
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#define SLEEP_MS(ms) Sleep(ms)
#else
#include <unistd.h>
#include <sys/time.h>
#include <termios.h>
#include <fcntl.h>
#define SLEEP_MS(ms) usleep((ms) * 1000)
#endif

static volatile int keep_running = 1;

void handle_sigint(int sig) {
    (void)sig;
    keep_running = 0;
}

#ifdef _WIN32
void enable_terminal_features(void) {
    /* Set console to UTF-8 output */
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) return;
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
}

/* Non-blocking key check supporting both Console and MSYS2/Git-Bash mintty */
int check_keypress(void) {
    HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
    DWORD bytesAvail = 0;
    if (PeekNamedPipe(hIn, NULL, 0, NULL, &bytesAvail, NULL) && bytesAvail > 0) {
        char ch = 0;
        DWORD bytesRead = 0;
        if (ReadFile(hIn, &ch, 1, &bytesRead, NULL) && bytesRead > 0) {
            return (unsigned char)ch;
        }
    }
    if (_kbhit()) {
        return _getch();
    }
    return -1;
}
#else
static struct termios orig_termios;

void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

void enable_terminal_features(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}

int check_keypress(void) {
    unsigned char ch;
    if (read(STDIN_FILENO, &ch, 1) == 1) {
        return ch;
    }
    return -1;
}
#endif

double get_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

/* ── Pattern Stamp Helpers ─────────────────────────────────────────────── */
void randomize_grid(unsigned char *grid, int W, int H, unsigned int *seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    srand(*seed);
    for (int i = 0; i < W * H; i++) {
        grid[i] = (rand() % 100 < 30) ? 1 : 0;
    }
}

void spawn_glider(unsigned char *grid, int W, int H, int x, int y) {
    int coords[5][2] = {{1,0}, {2,1}, {0,2}, {1,2}, {2,2}};
    for (int i = 0; i < 5; i++) {
        int gx = (x + coords[i][0] + W) % W;
        int gy = (y + coords[i][1] + H) % H;
        grid[gy * W + gx] = 1;
    }
}

void spawn_pulsar(unsigned char *grid, int W, int H, int cx, int cy) {
    int p[48][2] = {
        {2,0},{3,0},{4,0},{8,0},{9,0},{10,0},
        {0,2},{5,2},{7,2},{12,2},{0,3},{5,3},{7,3},{12,3},{0,4},{5,4},{7,4},{12,4},
        {2,5},{3,5},{4,5},{8,5},{9,5},{10,5},
        {2,7},{3,7},{4,7},{8,7},{9,7},{10,7},
        {0,8},{5,8},{7,8},{12,8},{0,9},{5,9},{7,9},{12,9},{0,10},{5,10},{7,10},{12,10},
        {2,12},{3,12},{4,12},{8,12},{9,12},{10,12}
    };
    for (int i = 0; i < 48; i++) {
        int gx = (cx - 6 + p[i][0] + W) % W;
        int gy = (cy - 6 + p[i][1] + H) % H;
        grid[gy * W + gx] = 1;
    }
}

void spawn_gosper_gun(unsigned char *grid, int W, int H, int ox, int oy) {
    int p[36][2] = {
        {24,0},{22,1},{24,1},{12,2},{13,2},{20,2},{21,2},{34,2},{35,2},
        {11,3},{15,3},{20,3},{21,3},{34,3},{35,3},{0,4},{1,4},{10,4},{16,4},
        {20,4},{21,4},{0,5},{1,5},{10,5},{14,5},{16,5},{17,5},{22,5},{24,5},
        {10,6},{16,6},{24,6},{11,7},{15,7},{12,8},{13,8}
    };
    for (int i = 0; i < 36; i++) {
        int gx = (ox + p[i][0] + W) % W;
        int gy = (oy + p[i][1] + H) % H;
        grid[gy * W + gx] = 1;
    }
}

/* ── Main Simulation Routine ───────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    int W        = (argc > 1) ? atoi(argv[1]) : 64;
    int H        = (argc > 2) ? atoi(argv[2]) : 32;
    int delay_ms = (argc > 3) ? atoi(argv[3]) : 50;
    unsigned int seed = (argc > 4) ? (unsigned int)atoi(argv[4]) : 42;

    if (W < 10) W = 64;
    if (H < 10) H = 32;
    if (delay_ms < 5) delay_ms = 5;

    signal(SIGINT, handle_sigint);
    enable_terminal_features();

    /* Allocate grid buffers */
    unsigned char *grid_a = (unsigned char *)calloc(W * H, sizeof(unsigned char));
    unsigned char *grid_b = (unsigned char *)calloc(W * H, sizeof(unsigned char));
    if (!grid_a || !grid_b) {
        fprintf(stderr, "Error: Memory allocation failed for grid %dx%d\n", W, H);
        return 1;
    }

    unsigned char *current = grid_a;
    unsigned char *next    = grid_b;

    randomize_grid(current, W, H, &seed);

    /* Hide cursor and clear screen */
    printf("\033[?25l\033[2J\033[H");
    fflush(stdout);

    /* Allocate frame string buffer */
    int buf_size = (W + 16) * (H + 16) * 16;
    char *out_buf = (char *)malloc(buf_size);
    if (!out_buf) {
        fprintf(stderr, "Buffer allocation failed\n");
        return 1;
    }

    int generation = 0;
    int is_paused = 0;
    int step_once = 0;
    char status_msg[64] = "RUNNING";

    double t_last_fps = get_time_sec();
    int frames_fps = 0;
    double current_fps = 0.0;

    while (keep_running) {
        /* Check keyboard inputs without blocking */
        int key = check_keypress();
        if (key != -1) {
            if (key == ' ' ) {
                is_paused = !is_paused;
                strcpy(status_msg, is_paused ? "PAUSED" : "RUNNING");
            } else if (key == 's' || key == 'S') {
                is_paused = 1;
                step_once = 1;
                strcpy(status_msg, "STEPPED");
            } else if (key == 'r' || key == 'R') {
                randomize_grid(current, W, H, &seed);
                generation = 0;
                strcpy(status_msg, "RANDOMIZED");
            } else if (key == 'c' || key == 'C') {
                memset(current, 0, W * H);
                generation = 0;
                strcpy(status_msg, "CLEARED");
            } else if (key == '+' || key == '=') {
                if (delay_ms > 10) delay_ms -= 10;
                else if (delay_ms > 2) delay_ms -= 2;
                sprintf(status_msg, "DELAY: %dms", delay_ms);
            } else if (key == '-' || key == '_') {
                delay_ms += 10;
                if (delay_ms > 500) delay_ms = 500;
                sprintf(status_msg, "DELAY: %dms", delay_ms);
            } else if (key == '1') {
                spawn_glider(current, W, H, W / 2, H / 2);
                strcpy(status_msg, "GLIDER SPAWNED");
            } else if (key == '2') {
                spawn_pulsar(current, W, H, W / 2, H / 2);
                strcpy(status_msg, "PULSAR SPAWNED");
            } else if (key == '3') {
                spawn_gosper_gun(current, W, H, 2, 2);
                strcpy(status_msg, "GOSPER GUN SPAWNED");
            } else if (key == 'q' || key == 'Q' || key == 27) {
                keep_running = 0;
                break;
            }
        }

        /* Evolve grid if running or stepping */
        int live_count = 0;
        if (!is_paused || step_once) {
            generation++;
            step_once = 0;

            for (int y = 0; y < H; y++) {
                int y_up   = ((y == 0) ? H - 1 : y - 1) * W;
                int y_curr = y * W;
                int y_down = ((y == H - 1) ? 0 : y + 1) * W;

                for (int x = 0; x < W; x++) {
                    int x_left  = (x == 0) ? W - 1 : x - 1;
                    int x_right = (x == W - 1) ? 0 : x + 1;

                    int neighbors = current[y_up + x_left]   + current[y_up + x]   + current[y_up + x_right] +
                                    current[y_curr + x_left]                       + current[y_curr + x_right] +
                                    current[y_down + x_left] + current[y_down + x] + current[y_down + x_right];

                    unsigned char c = current[y_curr + x];
                    unsigned char res = (c == 1) ? ((neighbors == 2 || neighbors == 3) ? 1 : 0)
                                                 : ((neighbors == 3) ? 1 : 0);
                    next[y_curr + x] = res;
                    live_count += res;
                }
            }

            /* Swap pointers */
            unsigned char *tmp = current;
            current = next;
            next = tmp;
        } else {
            /* If paused, count existing alive cells */
            for (int i = 0; i < W * H; i++) {
                live_count += current[i];
            }
        }

        /* Compute framerate */
        frames_fps++;
        double now = get_time_sec();
        if (now - t_last_fps >= 0.4) {
            current_fps = (double)frames_fps / (now - t_last_fps);
            frames_fps = 0;
            t_last_fps = now;
        }

        /* Render frame into out_buf */
        char *ptr = out_buf;
        ptr += sprintf(ptr, "\033[H"); /* Cursor to top-left */

        /* Header Border */
        ptr += sprintf(ptr, "\033[1;36m+");
        for (int i = 0; i < W + 2; i++) ptr += sprintf(ptr, "-");
        ptr += sprintf(ptr, "+\033[0m\n");

        /* Title Line */
        ptr += sprintf(ptr, "\033[1;36m|\033[1;32m  CONWAY'S GAME OF LIFE (LIVE INTERACTIVE)  \033[0m");
        int pad1 = (W + 2) - 44;
        for (int i = 0; i < (pad1 > 0 ? pad1 : 0); i++) ptr += sprintf(ptr, " ");
        ptr += sprintf(ptr, "\033[1;36m|\033[0m\n");

        /* Telemetry Line */
        ptr += sprintf(ptr, "\033[1;36m|\033[0m Gen:\033[1;33m%-5d\033[0m Alive:\033[1;32m%-5d(%.1f%%)\033[0m FPS:\033[1;35m%4.1f\033[0m [\033[1;37m%s\033[0m]",
                       generation, live_count, 100.0 * live_count / (W * H), current_fps, status_msg);
        int stats_len = 42 + (int)strlen(status_msg);
        int pad2 = (W + 2) - stats_len;
        for (int i = 0; i < (pad2 > 0 ? pad2 : 0); i++) ptr += sprintf(ptr, " ");
        ptr += sprintf(ptr, "\033[1;36m|\033[0m\n");

        /* Grid Separator */
        ptr += sprintf(ptr, "\033[1;36m+");
        for (int i = 0; i < W + 2; i++) ptr += sprintf(ptr, "-");
        ptr += sprintf(ptr, "+\033[0m\n");

        /* Grid View */
        for (int y = 0; y < H; y++) {
            ptr += sprintf(ptr, "\033[1;36m| \033[0m");
            for (int x = 0; x < W; x++) {
                if (current[y * W + x]) {
                    ptr += sprintf(ptr, "\033[1;32mO\033[0m");
                } else {
                    ptr += sprintf(ptr, "\033[90m.\033[0m");
                }
            }
            ptr += sprintf(ptr, "\033[1;36m |\033[0m\n");
        }

        /* Controls Footer */
        ptr += sprintf(ptr, "\033[1;36m+");
        for (int i = 0; i < W + 2; i++) ptr += sprintf(ptr, "-");
        ptr += sprintf(ptr, "+\033[0m\n");

        ptr += sprintf(ptr, "\033[1;33m [Space]\033[0m Pause/Play | \033[1;33m[S]\033[0m Step | \033[1;33m[R]\033[0m Reset | \033[1;33m[C]\033[0m Clear | \033[1;33m[Q]\033[0m Quit\n");
        ptr += sprintf(ptr, "\033[1;33m [+ / -]\033[0m Speed (%dms) | \033[1;33m[1]\033[0m Glider | \033[1;33m[2]\033[0m Pulsar | \033[1;33m[3]\033[0m Gun\n", delay_ms);

        fwrite(out_buf, 1, ptr - out_buf, stdout);
        fflush(stdout);

        SLEEP_MS(delay_ms);
    }

    /* Restore cursor and clear screen */
    printf("\033[?25h\n\033[1;32mSimulation stopped successfully.\033[0m\n");

    free(grid_a);
    free(grid_b);
    free(out_buf);
    return 0;
}
