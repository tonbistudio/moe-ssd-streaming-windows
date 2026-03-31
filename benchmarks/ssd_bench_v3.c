/*
 * ssd_bench_v3.c — Closing the gap with CrystalDiskMark
 *
 * Key changes from v2:
 *   1. Single shared file handle (one CreateFile, all threads use it)
 *   2. Pre-allocated OVERLAPPED + event pool (no alloc in hot path)
 *   3. Batch submit: fire all QD reads, then wait for all completions
 *   4. Try both OVERLAPPED events and IOCP completion models
 *   5. Test with FILE_FLAG_SEQUENTIAL_SCAN hint
 *   6. 1 MB block size (matches CDM default) aggregated into expert reads
 *
 * Compile: cl /O2 ssd_bench_v3.c /Fe:ssd_bench_v3.exe
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define FILE_SIZE_GB      4
#define MAX_THREADS       16
#define MAX_QD            256

static const int64_t FILE_SIZE = (int64_t)FILE_SIZE_GB << 30;

static double now_sec(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

/* ================================================================
 * Approach A: Event-based overlapped I/O (closest to CDM internals)
 *
 * Each thread owns a batch of OVERLAPPED structs with manual-reset
 * events. Submit all, then WaitForMultipleObjects on the batch.
 * ================================================================ */

typedef struct {
    HANDLE      hFile;       /* shared handle */
    int         thread_id;
    int         num_threads;
    int         queue_depth; /* outstanding I/Os per batch */
    int64_t     block_size;
    int         passes;
    int64_t     bytes_read;
    double      elapsed_sec;
} WorkerCtx;

static DWORD WINAPI worker_event(LPVOID arg) {
    WorkerCtx *ctx = (WorkerCtx *)arg;
    int qd = ctx->queue_depth;
    if (qd > MAX_QD) qd = MAX_QD;
    /* WaitForMultipleObjects limit is 64 */
    if (qd > 64) qd = 64;

    int64_t bsz = ctx->block_size;

    /* Pre-allocate buffers, overlapped structs, and events */
    char       **bufs = (char **)malloc(qd * sizeof(char *));
    OVERLAPPED  *ovs  = (OVERLAPPED *)calloc(qd, sizeof(OVERLAPPED));
    HANDLE      *evts = (HANDLE *)malloc(qd * sizeof(HANDLE));

    for (int i = 0; i < qd; i++) {
        bufs[i] = (char *)_aligned_malloc((size_t)bsz, 4096);
        evts[i] = CreateEvent(NULL, TRUE, FALSE, NULL);  /* manual reset */
    }

    /* This thread's stripe of the file */
    int64_t total_blocks   = FILE_SIZE / bsz;
    int64_t blocks_per_thr = total_blocks / ctx->num_threads;
    int64_t my_start       = blocks_per_thr * ctx->thread_id;

    int64_t bytes = 0;
    double t0 = now_sec();

    for (int pass = 0; pass < ctx->passes; pass++) {
        int64_t blk = 0;

        while (blk < blocks_per_thr) {
            /* How many to submit in this batch */
            int batch = qd;
            if (blk + batch > blocks_per_thr)
                batch = (int)(blocks_per_thr - blk);

            /* Submit batch */
            int pending = 0;
            for (int i = 0; i < batch; i++) {
                int64_t offset = (my_start + blk + i) * bsz;
                ResetEvent(evts[i]);
                memset(&ovs[i], 0, sizeof(OVERLAPPED));
                ovs[i].Offset     = (DWORD)(offset & 0xFFFFFFFF);
                ovs[i].OffsetHigh = (DWORD)(offset >> 32);
                ovs[i].hEvent     = evts[i];

                DWORD nread = 0;
                BOOL ok = ReadFile(ctx->hFile, bufs[i], (DWORD)bsz, &nread, &ovs[i]);
                if (ok) {
                    /* Completed synchronously */
                    bytes += nread;
                } else if (GetLastError() == ERROR_IO_PENDING) {
                    pending++;
                } else {
                    /* Error — skip */
                }
            }

            /* Wait for all pending */
            if (pending > 0) {
                /* Collect events for pending ops */
                HANDLE wait_evts[64];
                int    wait_idx[64];
                int    nwait = 0;
                for (int i = 0; i < batch; i++) {
                    /* Check if this one is still pending */
                    if (WaitForSingleObject(evts[i], 0) == WAIT_TIMEOUT) {
                        wait_evts[nwait] = evts[i];
                        wait_idx[nwait]  = i;
                        nwait++;
                    }
                }
                if (nwait > 0) {
                    WaitForMultipleObjects(nwait, wait_evts, TRUE, INFINITE);
                }
                /* Collect results */
                for (int i = 0; i < batch; i++) {
                    DWORD nread = 0;
                    if (GetOverlappedResult(ctx->hFile, &ovs[i], &nread, FALSE)) {
                        bytes += nread;
                    }
                }
            }

            blk += batch;
        }
    }

    ctx->elapsed_sec = now_sec() - t0;
    ctx->bytes_read  = bytes;

    for (int i = 0; i < qd; i++) {
        CloseHandle(evts[i]);
        _aligned_free(bufs[i]);
    }
    free(bufs); free(ovs); free(evts);
    return 0;
}

/* ================================================================
 * Approach B: IOCP with single shared handle
 * ================================================================ */

static DWORD WINAPI worker_iocp(LPVOID arg) {
    WorkerCtx *ctx = (WorkerCtx *)arg;
    int qd = ctx->queue_depth;
    if (qd > MAX_QD) qd = MAX_QD;
    int64_t bsz = ctx->block_size;

    /* Each thread creates its own IOCP associated with the shared handle.
       Actually, on Windows you can associate a handle with only ONE IOCP.
       So for this approach, we'll use GetOverlappedResultEx instead. */

    char       **bufs  = (char **)malloc(qd * sizeof(char *));
    OVERLAPPED  *ovs   = (OVERLAPPED *)calloc(qd, sizeof(OVERLAPPED));
    HANDLE      *evts  = (HANDLE *)malloc(qd * sizeof(HANDLE));
    int         *active = (int *)calloc(qd, sizeof(int));

    for (int i = 0; i < qd; i++) {
        bufs[i] = (char *)_aligned_malloc((size_t)bsz, 4096);
        evts[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
    }

    int64_t total_blocks   = FILE_SIZE / bsz;
    int64_t blocks_per_thr = total_blocks / ctx->num_threads;
    int64_t my_start       = blocks_per_thr * ctx->thread_id;

    int64_t bytes = 0;
    double t0 = now_sec();

    for (int pass = 0; pass < ctx->passes; pass++) {
        int64_t next_submit = 0;
        int64_t completed   = 0;

        /* Reset all slots */
        for (int i = 0; i < qd; i++) active[i] = 0;

        /* Main loop: keep slots filled, drain as they complete */
        while (completed < blocks_per_thr) {
            /* Submit into any free slots */
            for (int i = 0; i < qd && next_submit < blocks_per_thr; i++) {
                if (active[i]) continue;

                int64_t offset = (my_start + next_submit) * bsz;
                ResetEvent(evts[i]);
                memset(&ovs[i], 0, sizeof(OVERLAPPED));
                ovs[i].Offset     = (DWORD)(offset & 0xFFFFFFFF);
                ovs[i].OffsetHigh = (DWORD)(offset >> 32);
                ovs[i].hEvent     = evts[i];

                DWORD nread = 0;
                BOOL ok = ReadFile(ctx->hFile, bufs[i], (DWORD)bsz, &nread, &ovs[i]);
                if (ok) {
                    bytes += nread;
                    completed++;
                } else if (GetLastError() == ERROR_IO_PENDING) {
                    active[i] = 1;
                }
                next_submit++;
            }

            /* Build array of active events and wait for any */
            HANDLE wait_evts[64];
            int    wait_map[64];
            int    nwait = 0;
            for (int i = 0; i < qd && nwait < 64; i++) {
                if (active[i]) {
                    wait_map[nwait] = i;
                    wait_evts[nwait] = evts[i];
                    nwait++;
                }
            }
            if (nwait == 0) continue;

            DWORD ret = WaitForMultipleObjects(nwait, wait_evts, FALSE, INFINITE);
            if (ret >= WAIT_OBJECT_0 && ret < WAIT_OBJECT_0 + (DWORD)nwait) {
                /* Check all signaled events (there may be more than one) */
                for (int w = (int)(ret - WAIT_OBJECT_0); w < nwait; w++) {
                    if (WaitForSingleObject(wait_evts[w], 0) == WAIT_OBJECT_0) {
                        int slot = wait_map[w];
                        DWORD nread = 0;
                        GetOverlappedResult(ctx->hFile, &ovs[slot], &nread, FALSE);
                        bytes += nread;
                        completed++;
                        active[slot] = 0;
                    }
                }
            }
        }
    }

    ctx->elapsed_sec = now_sec() - t0;
    ctx->bytes_read  = bytes;

    for (int i = 0; i < qd; i++) {
        CloseHandle(evts[i]);
        _aligned_free(bufs[i]);
    }
    free(bufs); free(ovs); free(evts);
    return 0;
}

/* ================================================================ */

typedef DWORD (WINAPI *WorkerFn)(LPVOID);

static void run_config(const char *label, HANDLE hFile,
                       WorkerFn fn, int nthreads, int qd,
                       int block_kb, int passes) {
    WorkerCtx ctxs[MAX_THREADS];
    HANDLE    handles[MAX_THREADS];
    int64_t   bsz = (int64_t)block_kb * 1024;

    for (int i = 0; i < nthreads; i++) {
        ctxs[i] = (WorkerCtx){
            .hFile       = hFile,
            .thread_id   = i,
            .num_threads = nthreads,
            .queue_depth = qd,
            .block_size  = bsz,
            .passes      = passes,
        };
        handles[i] = CreateThread(NULL, 0, fn, &ctxs[i], 0, NULL);
    }

    WaitForMultipleObjects(nthreads, handles, TRUE, INFINITE);

    int64_t total_bytes = 0;
    double  max_time    = 0;
    for (int i = 0; i < nthreads; i++) {
        total_bytes += ctxs[i].bytes_read;
        if (ctxs[i].elapsed_sec > max_time)
            max_time = ctxs[i].elapsed_sec;
        CloseHandle(handles[i]);
    }

    double gb   = (double)total_bytes / (1 << 30);
    double gbps = max_time > 0 ? gb / max_time : 0;

    printf("  %-30s  %2dT  QD%-3d  %4dKB  %6.2f GB  %5.2fs  %6.2f GB/s\n",
           label, nthreads, qd, block_kb, gb, max_time, gbps);
}

static HANDLE open_test_file(const char *path, DWORD extra_flags) {
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                       OPEN_EXISTING,
                       FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING | extra_flags,
                       NULL);
}

int main(int argc, char **argv) {
    const char *path = "D:\\infra\\test_experts.bin";
    if (argc > 1) path = argv[1];

    printf("=== SSD Benchmark v3: Matching CrystalDiskMark ===\n");
    printf("Target: CDM shows 3.27 GB/s Seq Q8T1, we got 1.47 GB/s\n\n");

    /* Verify file exists */
    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        printf("Test file not found. Run ssd_bench.exe first.\n");
        return 1;
    }

    int passes = 3;

    /* ---- Test A: Event-based, CDM-style block sizes ---- */
    printf("--- Approach A: Event-based batch submit ---\n");
    printf("    (submit QD reads, WaitForMultipleObjects, repeat)\n\n");

    {
        HANDLE hf = open_test_file(path, 0);
        if (hf == INVALID_HANDLE_VALUE) {
            printf("Cannot open file: %lu\n", GetLastError());
            return 1;
        }

        /* CDM uses 1 MB blocks for sequential test */
        printf("  -- 1 MB blocks (CDM default) --\n");
        run_config("event 1MB", hf, worker_event, 1, 8,  1024, passes);
        run_config("event 1MB", hf, worker_event, 1, 32, 1024, passes);
        run_config("event 1MB", hf, worker_event, 2, 16, 1024, passes);
        run_config("event 1MB", hf, worker_event, 4, 8,  1024, passes);
        run_config("event 1MB", hf, worker_event, 8, 8,  1024, passes);

        printf("\n  -- 128 KB blocks --\n");
        run_config("event 128KB", hf, worker_event, 1, 32, 128, passes);
        run_config("event 128KB", hf, worker_event, 4, 16, 128, passes);
        run_config("event 128KB", hf, worker_event, 8, 8,  128, passes);

        printf("\n  -- 32 MB blocks (expert-sized) --\n");
        run_config("event 32MB", hf, worker_event, 1, 4,  32768, passes);
        run_config("event 32MB", hf, worker_event, 4, 4,  32768, passes);
        run_config("event 32MB", hf, worker_event, 8, 4,  32768, passes);

        CloseHandle(hf);
    }

    /* ---- Test B: Same but with SEQUENTIAL_SCAN hint ---- */
    printf("\n--- Approach A + FILE_FLAG_SEQUENTIAL_SCAN ---\n\n");
    {
        HANDLE hf = open_test_file(path, FILE_FLAG_SEQUENTIAL_SCAN);
        if (hf != INVALID_HANDLE_VALUE) {
            run_config("event+seqscan 1MB", hf, worker_event, 1, 8,  1024, passes);
            run_config("event+seqscan 1MB", hf, worker_event, 1, 32, 1024, passes);
            run_config("event+seqscan 1MB", hf, worker_event, 4, 8,  1024, passes);
            run_config("event+seqscan 1MB", hf, worker_event, 8, 8,  1024, passes);
            CloseHandle(hf);
        } else {
            printf("  (SEQUENTIAL_SCAN + NO_BUFFERING combination rejected)\n");
        }
    }

    /* ---- Test C: Pipeline approach (submit-drain-refill) ---- */
    printf("\n--- Approach B: Pipeline (submit/drain/refill) ---\n\n");
    {
        HANDLE hf = open_test_file(path, 0);
        if (hf != INVALID_HANDLE_VALUE) {
            run_config("pipeline 1MB", hf, worker_iocp, 1, 8,  1024, passes);
            run_config("pipeline 1MB", hf, worker_iocp, 1, 32, 1024, passes);
            run_config("pipeline 1MB", hf, worker_iocp, 4, 8,  1024, passes);
            run_config("pipeline 1MB", hf, worker_iocp, 8, 8,  1024, passes);
            run_config("pipeline 1MB", hf, worker_iocp, 1, 64, 1024, passes);
            CloseHandle(hf);
        }
    }

    /* ---- Test D: Per-thread file handles (eliminate contention) ---- */
    printf("\n--- Approach C: Per-thread handles + event batch ---\n");
    printf("    (each thread opens its own handle)\n\n");
    {
        /* For this test, worker opens its own handle */
        /* We'll use a wrapper that opens per-thread */
        /* Actually, let's just run the event approach with separate handles
           by opening multiple handles here */
        int cfgs[][2] = {{1,8}, {1,32}, {4,8}, {8,8}};
        for (int c = 0; c < 4; c++) {
            int nt = cfgs[c][0], qd = cfgs[c][1];
            WorkerCtx ctxs[MAX_THREADS];
            HANDLE handles[MAX_THREADS];
            HANDLE files[MAX_THREADS];

            for (int i = 0; i < nt; i++) {
                files[i] = open_test_file(path, 0);
                ctxs[i] = (WorkerCtx){
                    .hFile       = files[i],
                    .thread_id   = i,
                    .num_threads = nt,
                    .queue_depth = qd,
                    .block_size  = 1024 * 1024,
                    .passes      = passes,
                };
                handles[i] = CreateThread(NULL, 0, worker_event, &ctxs[i], 0, NULL);
            }
            WaitForMultipleObjects(nt, handles, TRUE, INFINITE);

            int64_t total_bytes = 0;
            double max_time = 0;
            for (int i = 0; i < nt; i++) {
                total_bytes += ctxs[i].bytes_read;
                if (ctxs[i].elapsed_sec > max_time)
                    max_time = ctxs[i].elapsed_sec;
                CloseHandle(handles[i]);
                CloseHandle(files[i]);
            }
            double gb = (double)total_bytes / (1 << 30);
            double gbps = max_time > 0 ? gb / max_time : 0;
            printf("  %-30s  %2dT  QD%-3d  1024KB  %6.2f GB  %5.2fs  %6.2f GB/s\n",
                   "per-thread handle 1MB", nt, qd, gb, max_time, gbps);
        }
    }

    printf("\n=== Compare your best number above against CDM's 3.27 GB/s ===\n");
    printf("If still ~1.5 GB/s: the bottleneck is in Windows file I/O overhead\n");
    printf("and CDM may use a lower-level storage driver path (DeviceIoControl).\n");

    return 0;
}
