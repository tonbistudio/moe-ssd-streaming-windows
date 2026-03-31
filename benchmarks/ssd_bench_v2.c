/*
 * ssd_bench_v2.c — Async I/O pipelined NVMe benchmark for Windows
 *
 * Key improvement over v1: uses I/O Completion Ports (IOCP) with deep
 * queue depth to keep the NVMe command queue saturated. This should
 * get much closer to the drive's rated sequential read speed.
 *
 * Compile: (via vcvarsall x64)
 *   cl /O2 ssd_bench_v2.c /Fe:ssd_bench_v2.exe
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ---------- tunables ---------- */
#define FILE_SIZE_GB      4
#define MAX_THREADS       16
#define MAX_QUEUE_DEPTH   128        /* outstanding I/Os per thread */

static const int64_t FILE_SIZE = (int64_t)FILE_SIZE_GB << 30;

/* ---------- high-res timer ---------- */
static double now_sec(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

/* ---------- one in-flight I/O slot ---------- */
typedef struct {
    OVERLAPPED ov;
    char      *buf;
    int        active;
} IOSlot;

/* ---------- per-thread context ---------- */
typedef struct {
    const char *path;
    int64_t     file_size;
    int64_t     chunk_size;
    int         thread_id;
    int         num_threads;
    int         queue_depth;
    int         use_nocache;
    int         pass_count;
    int64_t     bytes_read;
    double      elapsed_sec;
} ThreadCtx;

/* ---------- IOCP-based async reader ---------- */
static DWORD WINAPI reader_thread(LPVOID arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    int qd = ctx->queue_depth;
    if (qd > MAX_QUEUE_DEPTH) qd = MAX_QUEUE_DEPTH;

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

    /* Create IOCP and associate file handle */
    HANDLE hIOCP = CreateIoCompletionPort(hFile, NULL, 0, 1);
    if (!hIOCP) {
        fprintf(stderr, "[thread %d] CreateIoCompletionPort failed: %lu\n",
                ctx->thread_id, GetLastError());
        CloseHandle(hFile);
        return 1;
    }

    /* Allocate I/O slots */
    IOSlot *slots = (IOSlot *)calloc(qd, sizeof(IOSlot));
    for (int i = 0; i < qd; i++) {
        slots[i].buf = (char *)_aligned_malloc((size_t)ctx->chunk_size, 4096);
        slots[i].active = 0;
    }

    /* Calculate this thread's stripe */
    int64_t total_chunks = ctx->file_size / ctx->chunk_size;
    int64_t my_chunks    = total_chunks / ctx->num_threads;
    int64_t start_chunk  = my_chunks * ctx->thread_id;

    int64_t bytes = 0;
    double t0 = now_sec();

    for (int pass = 0; pass < ctx->pass_count; pass++) {
        int64_t next_submit = 0;   /* next chunk index to submit */
        int64_t completed   = 0;   /* chunks fully read */
        int     in_flight   = 0;

        /* Fill the pipeline */
        while (in_flight < qd && next_submit < my_chunks) {
            int slot_idx = in_flight;
            IOSlot *s = &slots[slot_idx];
            int64_t offset = (start_chunk + next_submit) * ctx->chunk_size;

            memset(&s->ov, 0, sizeof(OVERLAPPED));
            s->ov.Offset     = (DWORD)(offset & 0xFFFFFFFF);
            s->ov.OffsetHigh = (DWORD)(offset >> 32);
            s->active = 1;

            DWORD nread = 0;
            BOOL ok = ReadFile(hFile, s->buf, (DWORD)ctx->chunk_size, &nread, &s->ov);
            if (!ok && GetLastError() != ERROR_IO_PENDING) {
                fprintf(stderr, "[thread %d] ReadFile failed: %lu\n",
                        ctx->thread_id, GetLastError());
                s->active = 0;
                break;
            }
            if (ok) {
                /* completed synchronously */
                bytes += nread;
                completed++;
                s->active = 0;
            } else {
                in_flight++;
            }
            next_submit++;
        }

        /* Drain completions and submit new reads */
        while (completed < my_chunks) {
            DWORD nread = 0;
            ULONG_PTR key = 0;
            OVERLAPPED *pov = NULL;

            BOOL ok = GetQueuedCompletionStatus(hIOCP, &nread, &key, &pov, INFINITE);
            if (!ok) {
                DWORD err = GetLastError();
                if (pov) {
                    /* I/O error on a specific operation */
                    fprintf(stderr, "[thread %d] IOCP error: %lu\n",
                            ctx->thread_id, err);
                }
                break;
            }

            bytes += nread;
            completed++;
            in_flight--;

            /* Submit another read if we have more chunks */
            if (next_submit < my_chunks) {
                /* Reuse the completed slot */
                IOSlot *s = (IOSlot *)((char *)pov - offsetof(IOSlot, ov));
                int64_t offset = (start_chunk + next_submit) * ctx->chunk_size;

                memset(&s->ov, 0, sizeof(OVERLAPPED));
                s->ov.Offset     = (DWORD)(offset & 0xFFFFFFFF);
                s->ov.OffsetHigh = (DWORD)(offset >> 32);

                DWORD nr2 = 0;
                BOOL ok2 = ReadFile(hFile, s->buf, (DWORD)ctx->chunk_size, &nr2, &s->ov);
                if (!ok2 && GetLastError() != ERROR_IO_PENDING) {
                    fprintf(stderr, "[thread %d] ReadFile resubmit failed: %lu\n",
                            ctx->thread_id, GetLastError());
                } else if (ok2) {
                    bytes += nr2;
                    completed++;
                } else {
                    in_flight++;
                }
                next_submit++;
            }
        }
    }

    ctx->elapsed_sec = now_sec() - t0;
    ctx->bytes_read  = bytes;

    for (int i = 0; i < qd; i++)
        _aligned_free(slots[i].buf);
    free(slots);
    CloseHandle(hIOCP);
    CloseHandle(hFile);
    return 0;
}

/* ---------- run one config ---------- */
static void run_bench(const char *label, const char *path,
                      int num_threads, int queue_depth, int chunk_mb,
                      int passes, int nocache) {
    ThreadCtx ctxs[MAX_THREADS];
    HANDLE    handles[MAX_THREADS];
    int64_t   chunk_size = (int64_t)chunk_mb << 20;

    for (int i = 0; i < num_threads; i++) {
        ctxs[i] = (ThreadCtx){
            .path        = path,
            .file_size   = FILE_SIZE,
            .chunk_size  = chunk_size,
            .thread_id   = i,
            .num_threads = num_threads,
            .queue_depth = queue_depth,
            .use_nocache = nocache,
            .pass_count  = passes,
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

    double gb   = (double)total_bytes / (1 << 30);
    double gbps = gb / max_time;
    int total_qd = num_threads * queue_depth;

    printf("  %-22s  %2dT x QD%-3d = QD%-4d  %3dMB chunks  "
           "%6.2f GB in %5.2fs = %6.2f GB/s\n",
           label, num_threads, queue_depth, total_qd, chunk_mb,
           gb, max_time, gbps);
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    const char *path = "D:\\infra\\test_experts.bin";
    if (argc > 1) path = argv[1];

    printf("=== SSD Benchmark v2: Async IOCP Pipeline ===\n");
    printf("File: %s (%d GB)\n\n", path, FILE_SIZE_GB);

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    printf("Logical CPUs: %lu\n", si.dwNumberOfProcessors);

    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    printf("Physical RAM: %.1f GB\n\n", (double)ms.ullTotalPhys / (1 << 30));

    /* Check test file exists */
    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        printf("Test file not found. Run ssd_bench.exe first to create it.\n");
        return 1;
    }

    /*
     * Phase 1: Raw SSD — sweep queue depth and chunk size
     * The hypothesis: v1 was slow because QD=1 per thread.
     * NVMe drives need QD>=4 to saturate.
     */
    printf("--- Phase 1: Raw SSD — Queue Depth Sweep ---\n");
    printf("    (FILE_FLAG_NO_BUFFERING, bypasses page cache)\n\n");

    /* Fixed 4 threads, sweep queue depth */
    int qd_list[] = {1, 2, 4, 8, 16, 32, 64};
    int n_qd = sizeof(qd_list) / sizeof(qd_list[0]);
    for (int i = 0; i < n_qd; i++) {
        run_bench("raw (QD sweep)", path, 4, qd_list[i], 32, 3, 1);
    }

    /* Fixed QD=16, sweep chunk sizes */
    printf("\n--- Phase 2: Raw SSD — Chunk Size Sweep (4T, QD16) ---\n\n");
    int chunk_list[] = {1, 4, 8, 16, 32, 64, 128};
    int n_chunk = sizeof(chunk_list) / sizeof(chunk_list[0]);
    for (int i = 0; i < n_chunk; i++) {
        run_bench("raw (chunk sweep)", path, 4, 16, chunk_list[i], 3, 1);
    }

    /* Sweep thread count with good QD and chunk */
    printf("\n--- Phase 3: Raw SSD — Thread Sweep (QD16, 32MB) ---\n\n");
    int t_list[] = {1, 2, 4, 8};
    int n_t = sizeof(t_list) / sizeof(t_list[0]);
    for (int i = 0; i < n_t; i++) {
        run_bench("raw (thread sweep)", path, t_list[i], 16, 32, 3, 1);
    }

    /* Best raw config: find empirically from above, but also test a big config */
    printf("\n--- Phase 4: Raw SSD — Max Effort ---\n\n");
    run_bench("raw MAX", path, 4, 32, 32, 5, 1);
    run_bench("raw MAX", path, 8, 16, 32, 5, 1);
    run_bench("raw MAX", path, 2, 64, 64, 5, 1);
    run_bench("raw MAX", path, 1, 128, 128, 5, 1);

    /* Page cache */
    printf("\n--- Phase 5: Page Cache — Best Configs ---\n\n");
    printf("  Pre-warming...\n");
    run_bench("(warmup)", path, 8, 16, 32, 1, 0);
    printf("\n");

    run_bench("cache", path, 4, 16, 32, 5, 0);
    run_bench("cache", path, 8, 16, 32, 5, 0);
    run_bench("cache", path, 4, 32, 32, 5, 0);
    run_bench("cache", path, 2, 32, 64, 5, 0);

    printf("\n=== v1 vs v2 comparison will show in the numbers above ===\n");
    printf("Look for the highest raw SSD number — that's your true NVMe limit.\n");
    printf("The page cache numbers show your memory bandwidth ceiling.\n");

    return 0;
}
