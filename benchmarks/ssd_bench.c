/*
 * ssd_bench.c — Multi-threaded positioned read benchmark for Windows NVMe
 *
 * Measures parallel ReadFile + OVERLAPPED throughput to validate
 * whether an SSD can support MoE expert streaming.
 *
 * Compile: cl /O2 ssd_bench.c /link /out:ssd_bench.exe
 *    -or-  gcc -O2 -o ssd_bench.exe ssd_bench.c
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ---------- tunables ---------- */
#define FILE_SIZE_GB      4          /* test file size */
#define CHUNK_SIZE_MB     32         /* per-read chunk — simulates one MoE expert */
#define MAX_THREADS       16
#define WARMUP_PASSES     2
#define BENCH_PASSES      5

static const int64_t FILE_SIZE  = (int64_t)FILE_SIZE_GB << 30;
static const int64_t CHUNK_SIZE = (int64_t)CHUNK_SIZE_MB << 20;

/* ---------- per-thread context ---------- */
typedef struct {
    const char *path;
    int64_t     file_size;
    int64_t     chunk_size;
    int         thread_id;
    int         num_threads;
    int         pass_count;
    int         use_nocache;   /* FILE_FLAG_NO_BUFFERING — bypasses page cache */
    int64_t     bytes_read;
    double      elapsed_sec;
} ThreadCtx;

/* ---------- high-res timer ---------- */
static double now_sec(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

/* ---------- worker: positioned reads across a stripe of the file ---------- */
static DWORD WINAPI reader_thread(LPVOID arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;

    DWORD flags = FILE_FLAG_OVERLAPPED;
    if (ctx->use_nocache)
        flags |= FILE_FLAG_NO_BUFFERING;

    HANDLE hFile = CreateFileA(
        ctx->path, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, flags, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[thread %d] CreateFile failed: %lu\n",
                ctx->thread_id, GetLastError());
        return 1;
    }

    /* aligned buffer (4K alignment required for NO_BUFFERING) */
    char *buf = (char *)_aligned_malloc((size_t)ctx->chunk_size, 4096);
    if (!buf) { CloseHandle(hFile); return 1; }

    int64_t total_chunks = ctx->file_size / ctx->chunk_size;
    int64_t my_chunks    = total_chunks / ctx->num_threads;
    int64_t start_chunk  = my_chunks * ctx->thread_id;

    int64_t bytes = 0;
    double t0 = now_sec();

    for (int pass = 0; pass < ctx->pass_count; pass++) {
        for (int64_t c = 0; c < my_chunks; c++) {
            int64_t offset = (start_chunk + c) * ctx->chunk_size;

            OVERLAPPED ov;
            memset(&ov, 0, sizeof(ov));
            ov.Offset     = (DWORD)(offset & 0xFFFFFFFF);
            ov.OffsetHigh = (DWORD)(offset >> 32);
            ov.hEvent     = CreateEvent(NULL, TRUE, FALSE, NULL);

            DWORD nread = 0;
            BOOL ok = ReadFile(hFile, buf, (DWORD)ctx->chunk_size, &nread, &ov);
            if (!ok && GetLastError() == ERROR_IO_PENDING) {
                GetOverlappedResult(hFile, &ov, &nread, TRUE);
            }
            bytes += nread;
            CloseHandle(ov.hEvent);
        }
    }

    ctx->elapsed_sec = now_sec() - t0;
    ctx->bytes_read  = bytes;

    _aligned_free(buf);
    CloseHandle(hFile);
    return 0;
}

/* ---------- run one benchmark config ---------- */
static void run_bench(const char *label, const char *path,
                      int num_threads, int passes, int nocache) {
    ThreadCtx ctxs[MAX_THREADS];
    HANDLE    handles[MAX_THREADS];

    for (int i = 0; i < num_threads; i++) {
        ctxs[i] = (ThreadCtx){
            .path        = path,
            .file_size   = FILE_SIZE,
            .chunk_size  = CHUNK_SIZE,
            .thread_id   = i,
            .num_threads = num_threads,
            .pass_count  = passes,
            .use_nocache = nocache,
        };
        handles[i] = CreateThread(NULL, 0, reader_thread, &ctxs[i], 0, NULL);
    }

    WaitForMultipleObjects(num_threads, handles, TRUE, INFINITE);

    int64_t total_bytes = 0;
    double  max_time    = 0;
    for (int i = 0; i < num_threads; i++) {
        total_bytes += ctxs[i].bytes_read;
        if (ctxs[i].elapsed_sec > max_time)
            max_time = ctxs[i].elapsed_sec;
        CloseHandle(handles[i]);
    }

    double gb = (double)total_bytes / (1 << 30);
    double gbps = gb / max_time;

    printf("  %-28s  %2d threads  %6.2f GB in %5.2fs = %6.2f GB/s\n",
           label, num_threads, gb, max_time, gbps);
}

/* ---------- create test file ---------- */
static int create_test_file(const char *path) {
    printf("Creating %d GB test file: %s\n", FILE_SIZE_GB, path);
    printf("(this is one-time, will take a minute...)\n");

    HANDLE hFile = CreateFileA(path, GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot create test file: %lu\n", GetLastError());
        return 0;
    }

    /* write 64 MB at a time with pseudo-random data (avoid compression cheats) */
    const int BUF = 64 * 1024 * 1024;
    char *buf = (char *)malloc(BUF);
    srand(42);
    for (int i = 0; i < BUF; i++) buf[i] = (char)rand();

    int64_t written = 0;
    while (written < FILE_SIZE) {
        DWORD nw;
        int chunk = (FILE_SIZE - written < BUF) ? (int)(FILE_SIZE - written) : BUF;
        if (!WriteFile(hFile, buf, chunk, &nw, NULL)) {
            fprintf(stderr, "Write failed at offset %lld: %lu\n",
                    written, GetLastError());
            free(buf); CloseHandle(hFile);
            return 0;
        }
        written += nw;
        printf("\r  %.1f%%", 100.0 * written / FILE_SIZE);
        fflush(stdout);
    }
    printf("\r  done.                \n");

    free(buf);
    CloseHandle(hFile);
    return 1;
}

/* ---------- flush page cache for the file (best-effort) ---------- */
static void flush_cache(const char *path) {
    /* Open with NO_BUFFERING, then close — this is a hint to drop cached pages.
       Not perfect on Windows, but combined with reading 4GB that exceeds typical
       standby list pressure, it's reasonable. */
    printf("  (flushing page cache — best effort on Windows)\n");

    /* Write a large dummy allocation to pressure the standby list */
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);

    /* Use RAMMap-style flush: open the file unbuffered and read a byte */
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    const char *path = "D:\\infra\\test_experts.bin";
    if (argc > 1) path = argv[1];

    printf("=== SSD MoE Expert Streaming Benchmark ===\n");
    printf("File: %s  (%d GB, %d MB chunks)\n\n", path, FILE_SIZE_GB, CHUNK_SIZE_MB);

    /* system info */
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    printf("Logical CPUs: %lu\n", si.dwNumberOfProcessors);

    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    printf("Physical RAM: %.1f GB\n\n", (double)ms.ullTotalPhys / (1 << 30));

    /* create test file if needed */
    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        if (!create_test_file(path)) return 1;
    } else {
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, 0, NULL);
        LARGE_INTEGER sz;
        GetFileSizeEx(h, &sz);
        CloseHandle(h);
        if (sz.QuadPart < FILE_SIZE) {
            printf("Test file too small, recreating...\n");
            if (!create_test_file(path)) return 1;
        } else {
            printf("Using existing test file.\n");
        }
    }

    int thread_counts[] = {1, 2, 4, 8, 12};
    int n_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    /* ---- Phase 1: Raw SSD (bypass page cache) ---- */
    printf("\n--- Phase 1: Raw SSD (FILE_FLAG_NO_BUFFERING) ---\n");
    printf("    This bypasses the page cache to measure true NVMe speed.\n\n");

    for (int i = 0; i < n_configs; i++) {
        if (thread_counts[i] > MAX_THREADS) continue;
        /* warmup */
        run_bench("(warmup)", path, thread_counts[i], 1, 1);
        run_bench("raw SSD", path, thread_counts[i], BENCH_PASSES, 1);
    }

    /* ---- Phase 2: Page cache (warm reads) ---- */
    printf("\n--- Phase 2: Page Cache (warm reads) ---\n");
    printf("    Simulates hot MoE experts already cached in RAM.\n\n");

    /* pre-warm: read the whole file once with caching enabled */
    printf("  Pre-warming page cache...\n");
    run_bench("(pre-warm)", path, 8, 1, 0);
    printf("\n");

    for (int i = 0; i < n_configs; i++) {
        if (thread_counts[i] > MAX_THREADS) continue;
        run_bench("page cache", path, thread_counts[i], BENCH_PASSES, 0);
    }

    /* ---- Phase 3: Mixed (simulates partial cache hits) ---- */
    printf("\n--- Phase 3: Mixed — cold start converging to warm ---\n");
    printf("    Simulates MoE inference: first pass cold, subsequent from cache.\n\n");

    flush_cache(path);
    for (int i = 0; i < n_configs; i++) {
        if (thread_counts[i] > MAX_THREADS) continue;
        /* 3 passes: first is cold(ish), rest warm up */
        run_bench("mixed (3 pass)", path, thread_counts[i], 3, 0);
    }

    /* ---- Summary ---- */
    printf("\n=== What This Means for MoE Expert Streaming ===\n");
    printf("Qwen3-MoE-30B: ~128 experts, ~200 MB each\n");
    printf("At 5 GB/s raw:  load 8 active experts in %.1f ms\n",
           8.0 * 200.0 / (5.0 * 1024.0) * 1000.0);
    printf("At 20 GB/s cache: load 8 active experts in %.1f ms\n",
           8.0 * 200.0 / (20.0 * 1024.0) * 1000.0);
    printf("At 40 GB/s cache: load 8 active experts in %.1f ms\n",
           8.0 * 200.0 / (40.0 * 1024.0) * 1000.0);
    printf("\nFor reference: typical token latency budget is 50-100 ms.\n");
    printf("If raw SSD >= ~3 GB/s with threading, your SABRENT can\n");
    printf("theoretically support expert streaming for smaller MoE models.\n");

    printf("\nDone. You can delete %s when finished.\n", path);
    return 0;
}
